import os
import threading
import yaml


_lock = threading.Lock()
_cache = {"mtime": None, "data": None}


def load_raw_config(config_path: str = "config/config.yaml") -> dict:
    if not os.path.exists(config_path):
        return {}
    mtime = os.path.getmtime(config_path)
    with _lock:
        if _cache["data"] is not None and _cache["mtime"] == mtime:
            return _cache["data"]
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            data = {}
        _cache["mtime"] = mtime
        _cache["data"] = data
        return data


def save_raw_config(data: dict, config_path: str = "config/config.yaml") -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with _lock:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        _cache["mtime"] = os.path.getmtime(config_path)
        _cache["data"] = data


def get_runtime_settings(config_path: str = "config/config.yaml") -> dict:
    cfg = load_raw_config(config_path)
    attendance_cfg = cfg.get("attendance", {}) if isinstance(cfg.get("attendance", {}), dict) else {}
    capture_cfg = cfg.get("capture", {}) if isinstance(cfg.get("capture", {}), dict) else {}
    perf_cfg = cfg.get("performance", {}) if isinstance(cfg.get("performance", {}), dict) else {}
    security_cfg = cfg.get("security", {}) if isinstance(cfg.get("security", {}), dict) else {}

    return {
        "attendance": {
            "dedup_seconds": int(attendance_cfg.get("dedup_seconds", 60)),
        },
        "capture": {
            "width": int(capture_cfg.get("width", 1280)),
            "height": int(capture_cfg.get("height", 720)),
            "frame_interval_ms": int(capture_cfg.get("frame_interval_ms", 33)),
            "jpeg_quality": float(capture_cfg.get("jpeg_quality", 0.7)),
        },
        "performance": {
            "max_inference_concurrency": int(perf_cfg.get("max_inference_concurrency", 2)),
            "max_ws_connections": int(perf_cfg.get("max_ws_connections", 32)),
        },
        "security": {
            "force_https": bool(security_cfg.get("force_https", False)),
        },
    }
