# -*- coding: utf-8 -*-
# Version: V2.0 (Latest)
# Description: 优化版后端，采用最小外接矩形（OBB）及像素级掩膜收缩算法。
# Pros: 精准贴合倾斜标注。

import os
import cv2
import numpy as np
import base64
import time
import json
import math
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from openai import OpenAI
from ultralytics import YOLO

# ================= 配置区域 =================
load_dotenv()  # 加载 .env 文件中的变量
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-vl-max"

# 类别映射：请根据你训练 YOLO 时的 classes.txt 确定 ID
TARGET_YOLO_CLASSES = {
    1: "平行度",
    2: "平面度",
    3: "圆度",
    4: "同轴度"

}
# ===========================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("正在初始化 PaddleOCR 模型...")
ocr_engine = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=True)

print("正在初始化 YOLO 符号检测模型...")
yolo_model = YOLO("best.pt")
print("模型全部初始化完成！")


# --- 辅助函数 ---

def fuse_ocr_yolo(ocr_polys, yolo_results):
    if not ocr_polys: return []
    fused_info = []
    yolo_boxes = yolo_results[0].boxes.data.cpu().numpy()
    used_yolo_indices = set()

    for ocr_poly in ocr_polys:
        ocr_poly = np.array(ocr_poly)
        ox_min, oy_min = np.min(ocr_poly, axis=0)
        ox_max, oy_max = np.max(ocr_poly, axis=0)
        oh = oy_max - oy_min
        o_cx, o_cy = (ox_min + ox_max) / 2, (oy_min + oy_max) / 2  # OCR 中心点

        best_ybox_idx = -1
        yolo_type = None

        for idx, ybox in enumerate(yolo_boxes):
            if idx in used_yolo_indices: continue
            yx1, yy1, yx2, yy2, yconf, ycls = ybox
            y_cx, y_cy = (yx1 + yx2) / 2, (yy1 + yy2) / 2  # YOLO 中心点

            # 1. 垂直方向重叠
            v_overlap = max(0, min(oy_max, yy2) - max(oy_min, yy1))
            # 2. 水平方向距离
            dist_x = ox_min - yx2
            # 3. 【新增：中心点欧式距离】
            center_dist = math.sqrt((o_cx - y_cx) ** 2 + (o_cy - y_cy) ** 2)

            # 严格匹配条件：
            # A. 直径模式：垂直重叠 > 50% 且 水平距离很近，且中心点距离不能超过高度的 3 倍
            is_match = False
            if v_overlap > 0.5 * oh and -15 < dist_x < oh * 1.0:
                if center_dist < oh * 2.5:  # 增加物理距离上限
                    is_match = True

            # B. 公差框模式：OCR 完全在 YOLO 内部
            elif ox_min >= yx1 and ox_max <= yx2 and oy_min >= yy1 and oy_max <= yy2:
                is_match = True

            if is_match:
                best_ybox_idx = idx
                yolo_type = TARGET_YOLO_CLASSES.get(int(ycls))
                break

        if best_ybox_idx != -1:
            used_yolo_indices.add(best_ybox_idx)
            ybox = yolo_boxes[best_ybox_idx]
            nx1, ny1 = min(ox_min, ybox[0]), min(oy_min, ybox[1])
            nx2, ny2 = max(ox_max, ybox[2]), max(oy_max, ybox[3])
            fused_results = np.array([[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]])
            fused_info.append({"poly": fused_results, "yolo_type": yolo_type})
        else:
            fused_info.append({"poly": ocr_poly, "yolo_type": None})

    return fused_info


def generate_detail_grid(original_img, polys, padding=40):
    if not polys: return None
    crops = []
    h_img, w_img = original_img.shape[:2]
    for i, poly in enumerate(polys):
        pts = np.array(poly, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        left_extra = int(h * 1.5)
        x_new, y_new = max(0, x - padding - left_extra), max(0, y - padding)
        w_new, h_new = min(w_img - x_new, w + 2 * padding + left_extra), min(h_img - y_new, h + 2 * padding)
        crop = original_img[y_new:y_new + h_new, x_new:x_new + w_new].copy()
        header_h = 35
        labeled_crop = np.ones((crop.shape[0] + header_h, crop.shape[1], 3), dtype=np.uint8) * 255
        labeled_crop[header_h:, :] = crop
        cv2.putText(labeled_crop, f"ID: {i + 1}", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        crops.append(labeled_crop)

    col_width = 300
    resized_crops = [cv2.resize(c, (col_width, int(c.shape[0] * (col_width / c.shape[1])))) for c in crops]
    cols = min(len(resized_crops), 3)
    rows_imgs = []
    for idx in range(0, len(resized_crops), cols):
        row_batch = resized_crops[idx:idx + cols]
        max_h = max(img.shape[0] for img in row_batch)
        row_canvas = np.ones((max_h, col_width * cols, 3), dtype=np.uint8) * 255
        for c_idx, c_img in enumerate(row_batch):
            row_canvas[0:c_img.shape[0], c_idx * col_width:(c_idx + 1) * col_width] = c_img
        rows_imgs.append(row_canvas)

    if not rows_imgs: return None
    final_grid = np.vstack(rows_imgs)
    return final_grid


def remove_blue_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array([100, 50, 50]), np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = img.copy()
    res[blue_mask > 0] = [255, 255, 255]
    return res


def min_dist_between_polys(poly1, poly2):
    b1, b2 = np.array(poly1), np.array(poly2)
    x1_min, y1_min = np.min(b1, axis=0)
    x1_max, y1_max = np.max(b1, axis=0)
    x2_min, y2_min = np.min(b2, axis=0)
    x2_max, y2_max = np.max(b2, axis=0)
    dx = max(0, x2_min - x1_max, x1_min - x2_max)
    dy = max(0, y2_min - y1_max, y1_min - y2_max)
    return np.sqrt(float(dx) ** 2 + float(dy) ** 2)


def refine_box_by_pixels(img_no_blue, poly, padding=5):
    """
    在原始区域略微扩大搜索，找到黑色像素的最小外接矩形
    """
    h_img, w_img = img_no_blue.shape[:2]
    pts = np.array(poly, dtype=np.int32)

    # 依然先用外接矩形切出 ROI 以提高速度
    x, y, w, h = cv2.boundingRect(pts)
    x_start, y_start = max(0, x - padding), max(0, y - padding)
    x_end, y_end = min(w_img, x + w + padding), min(h_img, y + h + padding)

    roi = img_no_blue[y_start:y_end, x_start:x_end]
    if roi.size == 0: return poly

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 二值化找到黑色文字部分
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    coords = cv2.findNonZero(binary)
    if coords is not None:
        # 关键修改：计算黑色像素的最小外接矩形而非轴对齐矩形
        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)

        # 将 ROI 相对坐标映射回原图全局坐标
        box[:, 0] += x_start
        box[:, 1] += y_start
        return box.astype(np.float32)

    return poly.astype(np.float32)


def merge_boxes_full(poly_list):
    """
    将多个多边形的点集汇总，计算它们的最小外接矩形（带旋转角度）
    """
    if not poly_list:
        return np.array([])

    # 将所有多边形的顶点拼接成一个长列表
    all_points = np.vstack(poly_list).astype(np.float32)

    # 使用 OpenCV 的 minAreaRect 计算最小面积外接矩形
    # rect 结构为: ((中心x, 中心y), (宽, 高), 旋转角度)
    rect = cv2.minAreaRect(all_points)

    # 将 rect 转换为 4 个顶点的坐标
    box = cv2.boxPoints(rect)

    # 按照左上、右上、右下、左下的顺序稍微整理下点（可选，方便后续处理）
    # 这里直接返回 4 个点即可，cv2.polylines 会正确封闭
    return box.astype(np.float32)


def get_short_side_len(poly):
    p = np.array(poly)
    if len(p) < 3: return 10
    return min(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[1] - p[2]))


def get_poly_angle(poly):
    p = np.array(poly)
    if len(p) < 2: return 0.0
    dx, dy = p[1][0] - p[0][0], p[1][1] - p[0][1]
    angle_deg = math.degrees(math.atan2(dy, dx))
    if angle_deg < 0: angle_deg += 180
    return angle_deg


def analyze_with_qwenvl(img_base64, text_list):
    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

    # 构造更明确的 Prompt
    structured_texts = "\n".join([f"ID {i + 1}: {t}" for i, t in enumerate(text_list)])

    system_prompt = """你是一个CAD零件图纸分析专家。
    请分析图片中对应编号的尺寸标注。
    严格返回 JSON 数组格式，不要包含任何解释文字。
    格式示例：[{"id": 1, "type": "直径", "value": "10.5", "upper_tol": "", "lower_tol": ""}]
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    {"type": "text", "text": f"请识别以下 ID 区域的内容：\n{structured_texts}"}
                ]}
            ],
            temperature=0.01,
        )
        content = response.choices[0].message.content
        # 强力清理 JSON 标签
        content = content.replace("```json", "").replace("```", "").strip()
        return content
    except Exception as e:
        print(f"API 调用异常: {e}")
        return None


def filter_overlapping_polys(polys, texts, iom_threshold=0.8):
    """
    大鱼吃小鱼逻辑：如果一个小框 80% 的面积都在一个大框里，就删掉小框。
    """
    if not polys: return [], []

    n = len(polys)
    rects = []
    for p in polys:
        x, y, w, h = cv2.boundingRect(np.array(p, dtype=np.int32))
        rects.append([x, y, x + w, y + h, w * h])  # [x1, y1, x2, y2, area]

    to_remove = set()
    for i in range(n):
        for j in range(n):
            if i == j: continue

            bi, bj = rects[i], rects[j]
            # 计算交集矩形坐标
            x_inter1 = max(bi[0], bj[0])
            y_inter1 = max(bi[1], bj[1])
            x_inter2 = min(bi[2], bj[2])
            y_inter2 = min(bi[3], bj[3])

            if x_inter2 > x_inter1 and y_inter2 > y_inter1:
                inter_area = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
                # 计算交集占当前框 i 的比例 (IoM)
                iom = inter_area / bi[4]

                # 如果 i 被 j 包含（i 面积更小且交集占比高）
                if iom > iom_threshold and bi[4] < bj[4]:
                    to_remove.add(i)
                # 特殊情况：如果两个框面积几乎一样且重叠度极高，删掉 text 长度短的那个
                elif iom > 0.9 and abs(bi[4] - bj[4]) < (0.1 * bi[4]):
                    if len(texts[i]) < len(texts[j]):
                        to_remove.add(i)

    return [polys[k] for k in range(n) if k not in to_remove], \
        [texts[k] for k in range(n) if k not in to_remove]


# --- 核心处理逻辑 ---
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    logs = []

    def log(msg):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}");
        print(logs[-1])

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        opencv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if opencv_img is None: return {"error": "无效图片"}

        process_img = remove_blue_color(opencv_img)
        log("开始 OCR 推理...")
        results = ocr_engine.predict(input=process_img)
        r = results[0] if results else None

        log("开始 YOLO 推理...")
        yolo_res = yolo_model.predict(process_img, conf=0.2, imgsz=640)

        keep_texts, keep_polys = [], []
        if r:
            raw_texts, raw_polys = r['rec_texts'], r['dt_polys']
            n = len(raw_texts)
            parent = list(range(n))

            def find(i):
                if parent[i] == i: return i
                parent[i] = find(parent[i]);
                return parent[i]

            def union(i, j):
                root_i, root_j = find(i), find(j)
                if root_i != root_j: parent[root_j] = root_i

            for i in range(n):
                for j in range(i + 1, n):
                    dist = min_dist_between_polys(raw_polys[i], raw_polys[j])
                    h_i = get_short_side_len(raw_polys[i])

                    # 1. 计算连线角度
                    p1 = np.mean(raw_polys[i], axis=0)
                    p2 = np.mean(raw_polys[j], axis=0)
                    line_angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180

                    # 2. 获取文字框自身的旋转角度
                    box_angle = get_poly_angle(raw_polys[i])

                    # 3. 角度差
                    angle_diff = abs(line_angle - box_angle)
                    #if angle_diff > 90: angle_diff = 180 - angle_diff

                    # 【核心优化】
                    # 只有当两个碎片在“文字行进方向”上且距离极近（5%高度）才合并
                    # 这样 ID 8 和 9 虽然物理距离近，但它们是“并排”而不是“顺着”排的，角度差会很大
                    if angle_diff < 15:  # 顺着文字方向
                        threshold = h_i * 0.1  # 允许 10% 的空隙
                    else:  # 跨行方向（如并排的垂直标注）
                        threshold = h_i * 0.01  # 几乎不允许任何空隙

                    if dist < threshold:
                        union(i, j)

            grouped = {}
            for i in range(n):
                root = find(i);
                grouped.setdefault(root, []).append(i)

            for root in grouped:
                indices = sorted(grouped[root], key=lambda idx: raw_polys[idx][0][0])
                keep_texts.append("".join([raw_texts[i] for i in indices]))
                keep_polys.append(merge_boxes_full([raw_polys[i] for i in indices]))

            # 排序
            if keep_polys:
                combined = sorted(list(zip(keep_polys, keep_texts)), key=lambda x: np.mean(x[0], axis=0)[1])
                keep_polys, keep_texts = zip(*combined)
                keep_polys, keep_texts = list(keep_polys), list(keep_texts)
                log("正在执行重叠框清理 (大鱼吃小鱼)...")
                keep_polys, keep_texts = filter_overlapping_polys(keep_polys, keep_texts)

            # 融合 YOLO 并记录类型
            log("正在融合 YOLO 符号...")
            fused_info = fuse_ocr_yolo(keep_polys, yolo_res)
            keep_polys = [item["poly"] for item in fused_info]
            yolo_types_list = [item["yolo_type"] for item in fused_info]

            # 精调
            keep_polys = [refine_box_by_pixels(process_img, p,padding=2) for p in keep_polys]

        # 生成拼图
        collage_base64 = ""
        collage_img = generate_detail_grid(opencv_img, keep_polys)
        if collage_img is not None:
            _, buffer = cv2.imencode('.jpg', collage_img)
            collage_base64 = base64.b64encode(buffer).decode('utf-8')

        final_structured_data = []
        if keep_texts and collage_base64:
            log("正在调用 Aliyun Qwen-VL 分析...")
            ai_json = analyze_with_qwenvl(collage_base64, keep_texts)

            if ai_json:
                log(f"AI 原始返回内容预览: {ai_json[:100]}...")  # 打印前100个字符用于调试
                try:
                    ai_data = json.loads(ai_json)
                    # 灵活匹配 ID (防止 AI 返回 "1" 或 "ID 1")
                    ai_map = {}
                    for item in ai_data:
                        raw_id = str(item.get('id', ''))
                        clean_id = raw_id.replace("ID", "").replace("id", "").strip()
                        ai_map[clean_id] = item

                    exclude_types = ["文本", "说明", "比例", "剖视图", "注释", "几何公差", "标题", "材料", "硬度测试区", "基准","标注"]

                    for i in range(len(keep_texts)):
                        cid = str(i + 1)
                        # 从 AI 结果里找，找不到就给个空字典
                        ai_item = ai_map.get(cid, {})

                        # 类别判定优先级：YOLO 建议 > AI 识别 > 默认
                        final_type = ai_item.get("type", "长度")

                        yolo_suggested = yolo_types_list[i]
                        if yolo_suggested in ["平行度","平面度","同轴度", "圆度"]:
                            final_type = yolo_suggested

                        if final_type in exclude_types: continue

                        final_structured_data.append({
                            "id": i + 1,
                            "type": final_type,
                            "value": ai_item.get("value", keep_texts[i]),
                            "upper_tol": ai_item.get("upper_tol", ""),
                            "lower_tol": ai_item.get("lower_tol", ""),
                            "points": keep_polys[i].tolist()
                        })
                except Exception as json_err:
                    log(f"JSON 解析失败! 错误原因: {json_err}")
                    # 解析失败时，至少保留 OCR 的原始值，不显示“未识别”
                    for i, text in enumerate(keep_texts):
                        final_structured_data.append(
                            {"id": i + 1, "type": yolo_types_list[i] or "OCR识别", "value": text,
                             "points": keep_polys[i].tolist()})
            else:
                log("AI 返回为空，进入兜底")
                # 同上


        # 绘图
        draw_img = opencv_img.copy()

        # --- 新增：画合并前的原始小框 (绿色) ---
        for poly in raw_polys:
            pts = np.array(poly).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(draw_img, [pts], True, (0, 255, 0), 2)  # 绿色细线

        for item in final_structured_data:
            pts = np.array(item['points']).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(draw_img, [pts], True, (0, 0, 255), 2)
            cv2.putText(draw_img, str(item['id']), (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)

        _, buffer_f = cv2.imencode('.jpg', draw_img)
        return {"status": "success",
                "processed_image": f"data:image/jpeg;base64,{base64.b64encode(buffer_f).decode('utf-8')}",
                "ai_result": json.dumps(final_structured_data, ensure_ascii=False)}

    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)