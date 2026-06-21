# 智慧课堂人脸识别考勤系统

**Smart Classroom Face Recognition Attendance System — 工业级 B/S 架构生物识别考勤管理平台**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![YOLO](https://img.shields.io/badge/YOLO-Face%20Detection-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

[English](#english) | [中文](#chinese)

---

<span id="chinese"></span>

## 项目简介

本项目是一个面向教育场景的**人脸识别考勤管理系统**，基于计算机视觉技术实现课堂考勤自动化。系统采用 FastAPI + SQLite + YOLO 检测 + ArcFace/CosFace/AdaFace 识别骨干网络的工业级 B/S 架构，支持实时视频流、模型热切换、加密存储和完整审计追踪。

### 核心功能

**实时考勤**
- 基于 WebSocket 的低延迟视频流传输，实时人脸检测与身份标注
- 自动打卡与去重逻辑（可配置时间窗口），实时计算今日出勤率
- 花名册过滤：仅已关联课程的学生计入考勤

**学生管理**
- 完整 CRUD 操作，支持 Excel 批量导入/导出
- 摄像头现场拍照注册，自动提取人脸嵌入向量
- 班级/学院维度筛选与管理

**课程与考勤任务**
- 课程管理（课程编号、教师、时间、地点、关联班级）
- 考勤任务生命周期管理：草稿 → 运行中 → 已关闭
- 课程-学生多对多关联

**数据分析**
- Chart.js 可视化图表展示出勤趋势与分布
- 低出勤率自动预警，缺勤名单一键查看
- 考勤记录导出为 Excel

**系统配置（热切换）**
- 检测模型（YOLO）与识别模型（ResNet/NexNet）在线切换，即时生效
- 相似度阈值、去重窗口、推理并发数等参数实时调整
- 模型签名校验：切换模型后旧嵌入向量自动标记失效并重新提取

**安全机制**
- JWT (HS256) Cookie 认证，30 分钟过期
- bcrypt 密码哈希
- Fernet 加密存储人脸图片与嵌入向量
- CSRF 中间件保护所有状态变更请求
- 完整审计日志（操作人、动作、资源、IP、结果）

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | Python 3.12, FastAPI, Uvicorn, Jinja2 |
| 数据库 | SQLite (WAL 模式, 外键约束, busy_timeout), SQLAlchemy 2.0 |
| 人脸检测 | YOLOv8/v9/v11/v12 Pose & Face 模型 (Ultralytics) |
| 人脸识别 | ResNet10, ResNet50, NexNet + ArcFace/CosFace/AdaFace 权重 |
| 前端 | HTML5, Bootstrap 5 (AdminLTE), Chart.js, FontAwesome |
| 安全 | python-jose (JWT), bcrypt, cryptography (Fernet), CSRF 中间件 |
| 异步 | asyncio, WebSocket, 基于队列的批量考勤写入 |
| 包管理 | UV (`uv.lock`) |

### 项目结构

```
systems/
├── config/
│   └── config.yaml              # 主配置文件
├── core/                        # 核心模块
│   ├── config_manager.py        # 线程安全配置加载器
│   ├── database.py              # SQLAlchemy 引擎与 SQLite 优化
│   ├── models.py                # ORM 模型（7 张表）
│   ├── security.py              # bcrypt 密码哈希
│   ├── crypto_manager.py        # Fernet 加解密（人脸数据）
│   ├── csrf.py                  # CSRF 中间件
│   ├── audit.py                 # 审计日志
│   ├── runtime_settings.py      # 热重载配置读取器
│   └── model_factory.py         # 骨干网络工厂
├── models/
│   ├── backbones/               # ResNet, NexNet
│   ├── detectors/               # YOLO 检测器封装
│   └── weights/                 # 模型权重文件
├── services/
│   └── face_service.py          # 人脸识别核心服务
├── web/
│   ├── app.py                   # FastAPI 主应用（路由/WebSocket/模板）
│   ├── templates/               # Jinja2 页面模板（13 个）
│   └── static/                  # CSS/JS 静态资源
├── scripts/                     # 工具脚本
├── data/faces/                  # 加密人脸图片存储
├── secrets/                     # 加密密钥（已 gitignore）
├── docs/                        # 项目文档
├── start.bat                    # Windows 启动脚本
└── pyproject.toml               # 项目配置
```

### 快速开始

**前置条件：** Python 3.10+, 推荐 CUDA GPU

**1. 克隆仓库**
```bash
git clone <repo-url>
cd systems
```

**2. 安装依赖**
```bash
uv sync
# 或 pip install -r requirements.txt
```

**3. 准备模型权重**

将 `.pt` 和 `.pth` 权重文件放入对应目录：
- 检测模型：`models/weights/detection/`
- 识别模型：`models/weights/recognition/nexnet/` 或 `models/weights/recognition/resnet50/`

**4. 生成加密密钥**
```bash
python scripts/generate_face_data_key.py
```

**5. 启动服务**
```bash
uvicorn web.app:app --host 127.0.0.1 --port 8000
# 或直接运行 start.bat
```

**6. 访问系统**

打开浏览器访问 `http://127.0.0.1:8000`
- 默认管理员：`admin` / `admin123`

### 配置说明

编辑 `config/config.yaml` 或通过 Web 设置页面在线修改：

```yaml
detector:
  model_path: models/weights/detection/yolo-swa.pt
  model_type: yolo
  conf_threshold: 0.1
  device: auto

recognition:
  backbone_type: nexnet           # resnet10 / resnet50 / nexnet
  weights_path: models/weights/recognition/nexnet/cosface.pth
  embedding_size: 512
  similarity_threshold: 0.75
  device: auto

preprocess:
  input_size: [112, 112]
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]

attendance:
  dedup_seconds: 60               # 同一学生同一课程的去重间隔

capture:
  width: 640
  height: 480
  frame_interval_ms: 200
  jpeg_quality: 0.6

performance:
  max_inference_concurrency: 6
  max_ws_connections: 32

security:
  force_https: false
```

### 系统角色

| 角色 | 权限 |
|------|------|
| **admin** | 全部功能：学生/课程/用户管理、考勤任务、系统设置、审计日志 |
| **teacher** | 实时监控、考勤任务管理、查看历史记录、我的课程 |
| **student** | 查看个人出勤记录 |

### 安全特性

- 人脸图片与嵌入向量使用 Fernet 对称加密存储（密钥来自环境变量或密钥文件）
- 所有 POST/PUT/DELETE 请求受 CSRF 中间件保护
- 密码经 bcrypt 哈希，JWT 令牌 30 分钟过期
- 完整审计日志覆盖所有关键操作
- 可选 HTTPS 强制与 HSTS

### 文档

- [数据库关系文档](docs/数据库关系.md)
- [安全与稳定性指南](docs/安全与稳定性.md)

---

<span id="english"></span>

## Overview

An industrial-grade face recognition attendance system for educational environments. Built with FastAPI + SQLite + YOLO detection + ArcFace/CosFace/AdaFace recognition backbones, featuring real-time video streaming, hot-swappable models, encrypted storage, and full audit trails.

### Key Features

- **Real-time Attendance**: WebSocket-based low-latency video streaming with real-time face detection and identity labeling. Auto check-in with dedup logic and roster filtering.
- **Student Management**: Full CRUD, Excel batch import/export, webcam face registration.
- **Course & Task Management**: Course scheduling, attendance task lifecycle (draft → running → closed), course-student many-to-many associations.
- **Data Analytics**: Chart.js visualizations, low-attendance warnings, one-click Excel export.
- **Hot-Swappable Models**: Switch detection/recognition models online via the Settings page. Model signature validation auto-invalidates stale embeddings.
- **Security**: JWT (HS256) auth, bcrypt hashing, Fernet face data encryption, CSRF middleware, full audit logging.

### Tech Stack

| Layer | Technology |
|-------|-------------|
| Backend | Python 3.12, FastAPI, Uvicorn, Jinja2 |
| Database | SQLite (WAL mode), SQLAlchemy 2.0 |
| Detection | YOLOv8/v9/v11/v12 Pose & Face (Ultralytics) |
| Recognition | ResNet10, ResNet50, NexNet with ArcFace/CosFace/AdaFace |
| Frontend | HTML5, Bootstrap 5 (AdminLTE), Chart.js, FontAwesome |
| Security | JWT, bcrypt, Fernet encryption, CSRF middleware |

### Quick Start

**Prerequisites:** Python 3.10+, CUDA GPU recommended

```bash
# Install dependencies
uv sync

# Generate encryption key
python scripts/generate_face_data_key.py

# Start server
uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` — default credentials: `admin` / `admin123`

### Configuration

Edit `config/config.yaml` or use the web Settings page:

```yaml
detector:
  model_path: models/weights/detection/yolo-swa.pt
  model_type: yolo
  conf_threshold: 0.1

recognition:
  backbone_type: nexnet
  weights_path: models/weights/recognition/nexnet/cosface.pth
  similarity_threshold: 0.75

attendance:
  dedup_seconds: 60

performance:
  max_inference_concurrency: 6
  max_ws_connections: 32
```

### License

MIT License
