"""
人脸识别脚本 —— 加载预注册的人脸数据，对 test/faces/ 目录下的图片进行识别并标注。

用法：python test/recognize.py
"""

import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 配置
# ============================================================
FACE_DATA_FILE = "test/face_data.pth"
RECOGNIZE_DIR = "test/faces"
OUTPUT_DIR = "test/output"
DET_THRESHOLD = 0.01
DET_IMGSZ = 1280
REC_THRESHOLD = 0.6
IOU_THRESHOLD = 0.4
MODEL_PATH = "models/weights/recognition/nexnet/arcface.pth"
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_CJK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
]


def _resolve_dir(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))


def _load_font(size: int = 36) -> ImageFont.FreeTypeFont:
    for fp in _CJK_FONT_CANDIDATES:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _iter_images(dir_path: str):
    for f in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            yield os.path.join(dir_path, f)


def _iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def _dedup_by_iou(results, threshold):
    if len(results) <= 1:
        return results
    sorted_results = sorted(results, key=lambda r: r["score"], reverse=True)
    keep = []
    for r in sorted_results:
        if all(_iou(r["box"], k["box"]) < threshold for k in keep):
            keep.append(r)
    return keep


def draw_annotations(image_bgr, faces, font):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    for face in faces:
        x1, y1, x2, y2 = face["box"]
        name = face["name"]
        score = face["score"]

        is_known = name != "Unknown"
        box_color = (0, 220, 0) if is_known else (220, 0, 0)
        label_bg = (0, 200, 0) if is_known else (200, 0, 0)

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=6)

        text = f"{name} [{score:.2f}]"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        label_y = y1 - th - 10
        if label_y < 0:
            label_y = y2 + 10

        pad_x, pad_y = 12, 6
        draw.rectangle(
            [x1, label_y, x1 + tw + pad_x, label_y + th + pad_y],
            fill=label_bg,
        )
        draw.text((x1 + pad_x // 2, label_y + pad_y // 2), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    face_data_file = _resolve_dir(FACE_DATA_FILE)
    recognize_dir = _resolve_dir(RECOGNIZE_DIR)
    output_dir = _resolve_dir(OUTPUT_DIR)

    if not os.path.isfile(face_data_file):
        print(f"[错误] 人脸数据文件不存在: {face_data_file}")
        print(f"请先运行 python test/register.py 注册人脸")
        return

    if not os.path.isdir(recognize_dir):
        print(f"[错误] 识别目录不存在: {recognize_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---- 加载预注册的人脸数据 ----
    data = torch.load(face_data_file, map_location="cpu", weights_only=False)
    known_faces = data["known_faces"]
    student_labels = data["student_labels"]
    print(f"已加载 {len(known_faces)} 个注册人脸")

    # ---- 初始化服务 ----
    from services.face_service import FaceRecognitionService
    FaceRecognitionService._load_faces_from_db = lambda self: None

    from core.config_manager import Config
    cfg = Config()
    cfg._cfg.setdefault("recognition", {})["weights_path"] = os.path.normpath(
        os.path.join(PROJECT_ROOT, MODEL_PATH)
    )

    service = FaceRecognitionService(config=cfg)
    service.detector.conf_threshold = DET_THRESHOLD
    service.similarity_threshold = REC_THRESHOLD
    service.detector.model.overrides["imgsz"] = DET_IMGSZ

    # 将预注册的人脸数据注入 service
    service.known_faces = {sid: feat.to(service.device) for sid, feat in known_faces.items()}
    service.student_labels = dict(student_labels)

    # 两阶段检测补丁
    _original_detect_faces = service.detector.detect_faces
    _REF_RES = 640

    def _patched_detect_faces(image_path_or_array, align=True, output_size=112):
        detections = _original_detect_faces(image_path_or_array, align=False, output_size=output_size)
        if not align or not detections:
            return detections

        if isinstance(image_path_or_array, np.ndarray):
            img = image_path_or_array
        else:
            img = cv2.imread(image_path_or_array)
        if img is None:
            return detections
        h, w = img.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            margin_w, margin_h = int(bw * 0.2), int(bh * 0.2)
            cx1 = max(0, x1 - margin_w)
            cy1 = max(0, y1 - margin_h)
            cx2 = min(w, x2 + margin_w)
            cy2 = min(h, y2 + margin_h)
            face_crop = img[cy1:cy2, cx1:cx2]
            if face_crop.size == 0:
                continue

            crop_h, crop_w = face_crop.shape[:2]
            scale = _REF_RES / max(crop_w, crop_h)
            if scale != 1.0:
                face_crop = cv2.resize(face_crop,
                    (max(1, int(crop_w * scale)), max(1, int(crop_h * scale))),
                    interpolation=cv2.INTER_CUBIC)

            saved_imgsz = service.detector.model.overrides.get("imgsz")
            service.detector.model.overrides["imgsz"] = _REF_RES
            try:
                sub_dets = _original_detect_faces(face_crop, align=True, output_size=output_size)
            finally:
                if saved_imgsz is not None:
                    service.detector.model.overrides["imgsz"] = saved_imgsz

            if sub_dets and sub_dets[0].get("aligned_face") is not None:
                det["aligned_face"] = sub_dets[0]["aligned_face"]

        return detections

    service.detector.detect_faces = _patched_detect_faces

    # ---- 识别 ----
    print(f"\n{'识别结果':─^40}")
    font = _load_font(36)
    total_detected = 0
    total_matched = 0

    for img_path in _iter_images(recognize_dir):
        basename = os.path.basename(img_path)
        stem, ext = os.path.splitext(basename)

        image = cv2.imread(img_path)
        if image is None:
            print(f"  {basename}  无法读取")
            continue

        detections = service.detector.detect_faces(image, align=True, output_size=112)
        if not detections:
            print(f"  {basename}  未检测到人脸")
            continue

        face_features = []
        for det in detections:
            aligned = det.get("aligned_face")
            if aligned is None:
                face_features.append(None)
                continue
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            feat = service.extract_feature(Image.fromarray(rgb))
            face_features.append(feat)

        results = []
        for i, feat in enumerate(face_features):
            if feat is None:
                continue
            best_sid = None
            best_score = 0.0
            for sid, known_feat in service.known_faces.items():
                score = service.compute_similarity(feat, known_feat)
                if score > best_score:
                    best_score = score
                    best_sid = sid

            if best_score >= REC_THRESHOLD and best_sid is not None:
                results.append({
                    "box": detections[i]["box"],
                    "name": service.student_labels.get(best_sid, "Unknown"),
                    "score": float(best_score),
                })
            else:
                results.append({
                    "box": detections[i]["box"],
                    "name": "Unknown",
                    "score": float(best_score),
                })

        results = _dedup_by_iou(results, IOU_THRESHOLD)

        known_list = [(r["name"], r["score"]) for r in results if r["name"] != "Unknown"]
        unknown_list = [r["score"] for r in results if r["name"] == "Unknown"]
        n_total = len(results)
        n_known = len(known_list)
        total_detected += n_total
        total_matched += n_known

        print(f"  {basename}")
        print(f"    检测到 {n_total} 人  识别出 {n_known} 人  未知 {len(unknown_list)} 人")

        if known_list:
            names_str = " ".join(f"{n}[{s:.2f}]" for n, s in sorted(known_list, key=lambda x: -x[1]))
            print(f"    已知: {names_str}")
        if unknown_list:
            scores_str = " ".join(f"[{s:.2f}]" for s in sorted(unknown_list, reverse=True))
            print(f"    未知: {scores_str}")

        annotated = draw_annotations(image, results, font)
        out_path = os.path.join(output_dir, f"{stem}_annotated{ext}")
        cv2.imwrite(out_path, annotated)

    print(f"\n{'汇总':─^40}")
    print(f"  检测到 {total_detected} 人  识别出 {total_matched} 人  "
          f"识别率 {total_matched}/{total_detected} ({total_matched/total_detected*100:.0f}%)")
    print(f"  标注图已保存至 {output_dir}")


if __name__ == "__main__":
    main()
