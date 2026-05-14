"""
人脸注册脚本 —— 从 test/face/ 目录读取图片，提取人脸特征并保存到本地文件。

用法：python test/register.py
"""

import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# 配置
# ============================================================
REGISTER_DIR = "test/face"
OUTPUT_FILE = "test/face_data.pth"
DET_THRESHOLD = 0.01
DET_IMGSZ = 1280
MODEL_PATH = "models/weights/recognition/nexnet/arcface.pth"
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_dir(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))


def _iter_images(dir_path: str):
    for f in sorted(os.listdir(dir_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            yield os.path.join(dir_path, f)


def main():
    register_dir = _resolve_dir(REGISTER_DIR)
    output_file = _resolve_dir(OUTPUT_FILE)

    if not os.path.isdir(register_dir):
        print(f"[错误] 注册目录不存在: {register_dir}")
        return

    from services.face_service import FaceRecognitionService
    FaceRecognitionService._load_faces_from_db = lambda self: None

    from core.config_manager import Config
    cfg = Config()
    cfg._cfg.setdefault("recognition", {})["weights_path"] = os.path.normpath(
        os.path.join(PROJECT_ROOT, MODEL_PATH)
    )

    service = FaceRecognitionService(config=cfg)
    service.detector.conf_threshold = DET_THRESHOLD
    service.detector.model.overrides["imgsz"] = DET_IMGSZ

    _original_detect_faces = service.detector.detect_faces
    _REF_RES = 640

    def _patched_detect_faces(image_path_or_array, align=True, output_size=112):
        import cv2
        import numpy as np
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

    # ---- 注册 ----
    print(f"\n{'注册阶段':─^40}")
    known_faces = {}
    student_labels = {}
    next_id = 1000
    registered = 0
    failed = []

    for img_path in _iter_images(register_dir):
        name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            features = service.process_image(img_path)
            if not features:
                failed.append(name)
                continue
            feature = features[0].detach().cpu()
            known_faces[next_id] = feature
            student_labels[next_id] = name
            registered += 1
            print(f"  [OK] {name}  (id={next_id})")
            next_id += 1
        except Exception as e:
            failed.append(f"{name}({e})")

    print(f"\n  成功 {registered} 人", end="")
    if failed:
        print(f"  失败 {len(failed)}: {', '.join(failed)}", end="")
    print()

    if registered == 0:
        print("[错误] 未注册任何人脸，请检查 REGISTER_DIR 下是否有图片文件。")
        return

    # ---- 保存 ----
    torch.save({
        "known_faces": known_faces,
        "student_labels": student_labels,
        "model_sig": service.model_sig,
    }, output_file)
    print(f"\n人脸数据已保存至: {output_file}")


if __name__ == "__main__":
    main()
