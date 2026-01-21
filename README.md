# Smart Classroom Face Recognition System / 智慧课堂人脸识别考勤系统

**A robust, industrial-grade face recognition attendance system designed for educational environments.**
**专为教育环境设计的工业级人脸识别考勤系统。**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![YOLO](https://img.shields.io/badge/YOLO-Face%20Detection-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

[English](#-introduction) | [中文指南](#-项目简介)

---

<a name="-introduction"></a>
## 📖 Introduction

This project is a comprehensive **Biometric Attendance Management System** that leverages computer vision to automate classroom attendance. Unlike simple prototypes, this system is built with an "Industrial Grade" mindset, featuring a robust B/S architecture, database integration, and a professional Admin Dashboard.

It uses **YOLOv8/v11** for high-speed face detection and state-of-the-art recognition models (like **ArcFace/CosFace**) to ensure accuracy.

### ✨ Key Features

- **Real-time Monitoring**: Low-latency video streaming via WebSocket with real-time face bounding boxes and identity labels.
- **Student Management**: 
  - Complete CRUD operations.
  - **Batch Import**: Support for Excel (.xlsx) bulk student registration.
  - **Live Registration**: Capture photos directly using the webcam.
- **Smart Attendance**:
  - Automatic check-in logic with "Real-time Daily Attendance Rate" calculation.
  - **Class-based Management**: Filter and manage students by class.
- **Data Analytics**:
  - Visual charts for attendance trends and college/class distribution.
  - **Low Attendance Warnings**: Automatically flag students with poor attendance records.
  - **Export Reports**: One-click export of attendance logs to Excel.
- **System Configuration**:
  - **Hot-Swappable Models**: Switch between different Detection (YOLO) and Recognition models on the fly via the Settings page.
  - Adjustable similarity thresholds.
- **Course Management**: Schedule courses with precise date/time pickers.

### 🛠️ Tech Stack

- **Backend**: Python, FastAPI, SQLAlchemy, SQLite.
- **Computer Vision**: 
  - **Detection**: YOLOv8 / YOLOv11 (Pose/Face models).
  - **Recognition**: ArcFace / CosFace / AdaFace (via PyTorch).
- **Frontend**: HTML5, Bootstrap 5 (AdminLTE Theme), Chart.js, Jinja2 Templates.

---

<a name="-项目简介"></a>
## 📖 项目简介

本项目是一个综合性的**生物识别考勤管理系统**，利用计算机视觉技术实现课堂考勤自动化。与简单的原型不同，本系统基于“工业级”标准构建，采用稳健的 B/S 架构，集成数据库，并提供专业的后台管理仪表盘。

系统使用 **YOLOv8/v11** 进行高速人脸检测，并结合 **ArcFace/CosFace** 等先进识别模型以确保准确性。

### ✨ 核心功能

- **实时监控**：通过 WebSocket 实现低延迟视频流传输，实时显示人脸检测框和身份信息。
- **学生管理**：
  - 完整的增删改查（CRUD）操作。
  - **批量导入**：支持通过 Excel (.xlsx) 文件批量导入学生信息。
  - **现场注册**：支持使用摄像头直接拍照注册。
  - **班级管理**：以班级为核心进行筛选和管理。
- **智能考勤**：
  - 自动打卡逻辑，实时计算“今日实时出勤率”。
- **数据分析**：
  - 可视化图表展示出勤趋势和学院/班级分布。
  - **低出勤率预警**：自动标记出勤记录较差的学生。
  - **报表导出**：一键将考勤记录导出为 Excel 表格。
- **系统配置**：
  - **模型热切换**：通过设置页面动态切换不同的人脸检测（YOLO）和识别模型，即时生效。
  - 支持调整人脸比对的相似度阈值。
- **课程管理**：使用精确的日期/时间选择器安排课程。

### 🛠️ 技术栈

- **后端**: Python, FastAPI, SQLAlchemy, SQLite.
- **计算机视觉**: 
  - **检测**: YOLOv8 / YOLOv11 (Pose/Face models).
  - **识别**: ArcFace / CosFace / AdaFace (via PyTorch).
- **前端**: HTML5, Bootstrap 5 (AdminLTE 主题), Chart.js, Jinja2 模板引擎.

---

## 🚀 Getting Started / 快速开始

### Prerequisites / 前置条件

- Python 3.10+
- CUDA (Optional, for GPU acceleration / 可选，用于GPU加速)

### Installation / 安装步骤

1. **Clone the repository / 克隆仓库**
   ```bash
   git clone https://github.com/yourusername/Smart-Classroom-Face-Recognition.git
   cd Smart-Classroom-Face-Recognition
   ```

2. **Install Dependencies / 安装依赖**
   ```bash
   pip install -r requirements.txt
   # OR / 或者使用 uv
   uv pip install -r requirements.txt
   ```

3. **Prepare Models / 准备模型文件**
   Download the weight files (`.pt` and `.pth`) and place them in:
   请下载权重文件 (`.pt` 和 `.pth`) 并放置于：
   - `models/weights/detection/`
   - `models/weights/recognition/`

4. **Run the Server / 启动服务器**
   ```bash
   uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the Dashboard / 访问仪表盘**
   Open your browser and navigate to / 打开浏览器访问: `http://localhost:8000`
   - **Default Admin / 默认管理员**: `admin` / `admin123`

## ⚙️ Configuration / 配置

You can configure the system via the **Settings Page** in the web UI or by editing `config/config.yaml` manually.
您可以通过 Web 界面中的 **“模型配置”** 页面进行设置，或手动编辑 `config/config.yaml`。

```yaml
recognition:
  backbone_type: "resnet50"
  weights_path: "models/weights/recognition/CosFace/best.pth"
  similarity_threshold: 0.5

detector:
  model_type: "yolo"
  model_path: "models/weights/detection/yolo11n-pose.pt"
```

## 🔒 Privacy & Security / 隐私与安全

- **Data Privacy**: Student photos and database records are stored locally and added to `.gitignore` to prevent accidental leaks.
- **数据隐私**：学生照片和数据库记录仅存储在本地，并已添加到 `.gitignore` 中以防止意外泄露。
- **Authentication**: Secured with JWT (JSON Web Tokens) and Bcrypt password hashing.
- **认证安全**：使用 JWT 和 Bcrypt 密码哈希保护系统安全。

## 📄 License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
