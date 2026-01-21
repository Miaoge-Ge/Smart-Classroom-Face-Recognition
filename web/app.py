import sys
import os
import cv2
import numpy as np
import base64
import json
import pandas as pd
import io
import yaml
import glob
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from sqlalchemy import func

# Security imports
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from core.security import verify_password, get_password_hash

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.face_service import FaceRecognitionService
from core.database import engine, Base, get_db, SessionLocal
from core.models import Student, Attendance, AdminUser, Course
from core.config_manager import Config, global_config

# Create Tables
Base.metadata.create_all(bind=engine)

# Config
SECRET_KEY = "your-secret-key-keep-it-safe"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Global Service
service = None

def init_service():
    global service
    print("Initializing Face Service...")
    try:
        # Reload config
        new_config = Config()
        service = FaceRecognitionService(config=new_config)
        print("Face Service Initialized Successfully.")
    except Exception as e:
        print(f"Error initializing service: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init Default Admin if not exists
    db = SessionLocal()
    admin = db.query(AdminUser).filter(AdminUser.username == "admin").first()
    if not admin:
        hashed = get_password_hash("admin123")
        new_admin = AdminUser(username="admin", hashed_password=hashed, full_name="System Admin")
        db.add(new_admin)
        db.commit()
        print("Default admin created (admin/admin123)")
    db.close()
    
    init_service()
    
    yield
    print("Shutdown: Cleaning up...")

app = FastAPI(lifespan=lifespan, title="智慧课堂考勤管理系统")

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# --- Auth Helpers ---

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

async def login_required(request: Request):
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_307_TEMPORARY_REDIRECT, headers={"Location": "/login"})
    return user

# --- Pages ---

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_submit(request: Request, db: Session = Depends(get_db), username: str = Form(...), password: str = Form(...)):
    user = db.query(AdminUser).filter(AdminUser.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "用户名或密码错误"})
    
    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

@app.get("/dashboard")
async def dashboard_page(request: Request, user=Depends(login_required), db: Session = Depends(get_db)):
    # Statistics
    total_students = db.query(Student).count()
    total_courses = db.query(Course).count()
    
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_attendance = db.query(Attendance.student_id).filter(Attendance.timestamp >= today_start).distinct().count()
    
    # Calculate Real-time Rate (Today)
    # If total_students is 0, avoid division by zero
    if total_students > 0:
        weekly_rate = int((today_attendance / total_students) * 100)
    else:
        weekly_rate = 0
    
    # Chart Data (Last 7 days including today)
    dates = []
    counts = []
    # Loop from 6 days ago to today (0 days ago)
    for i in range(6, -1, -1):
        day = today_start - timedelta(days=i)
        next_day = day + timedelta(days=1)
        cnt = db.query(Attendance.student_id).filter(Attendance.timestamp >= day, Attendance.timestamp < next_day).distinct().count()
        dates.append(day.strftime("%m-%d"))
        counts.append(cnt)
        
    # Pie Chart (Colleges)
    colleges_stats = db.query(Student.college, func.count(Student.id)).group_by(Student.college).all()
    college_names = [c[0] if c[0] else "Unknown" for c in colleges_stats]
    college_counts = [c[1] for c in colleges_stats]
    
    stats = {
        "total_students": total_students,
        "today_attendance": today_attendance,
        "total_courses": total_courses,
        "weekly_rate": weekly_rate
    }
    
    chart_data = {
        "dates": dates,
        "counts": counts,
        "college_names": college_names,
        "college_counts": college_counts
    }
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "user": {"username": user},
        "stats": stats,
        "chart_data": chart_data,
        "title": "仪表盘"
    })

@app.get("/")
async def index(request: Request, user=Depends(login_required)):
    return templates.TemplateResponse("index.html", {"request": request, "user": {"username": user}, "title": "实时监控"})

@app.get("/students")
async def students_page(request: Request, class_name: str = None, user=Depends(login_required), db: Session = Depends(get_db)):
    query = db.query(Student)
    if class_name:
        query = query.filter(Student.class_name == class_name)
    students = query.all()
    
    classes = db.query(Student.class_name).distinct().all()
    classes = [c[0] for c in classes if c[0]]
    
    return templates.TemplateResponse("students.html", {
        "request": request, 
        "students": students, 
        "classes": classes,
        "current_class": class_name,
        "user": {"username": user},
        "title": "学生管理"
    })

@app.get("/history")
async def history_page(
    request: Request, 
    college: str = None, 
    search_query: str = None, 
    user=Depends(login_required),
    db: Session = Depends(get_db)
):
    query = db.query(Attendance).join(Student).order_by(Attendance.timestamp.desc())
    
    if college:
        query = query.filter(Student.college == college)
        
    if search_query:
        from sqlalchemy import or_
        query = query.filter(
            or_(
                Student.name.like(f"%{search_query}%"),
                Student.student_id == search_query
            )
        )
        
    records = query.limit(500).all()
    
    history_data = []
    for r in records:
        history_data.append({
            "name": r.student.name,
            "student_id": r.student.student_id or "-",
            "class_name": r.student.class_name or "-",
            "college": r.student.college or "-",
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "status": r.status
        })
        
    colleges = db.query(Student.college).distinct().all()
    colleges = [c[0] for c in colleges if c[0]]

    return templates.TemplateResponse("history.html", {
        "request": request, 
        "records": history_data, 
        "colleges": colleges,
        "current_college": college,
        "current_search": search_query,
        "user": {"username": user},
        "title": "考勤记录"
    })

@app.get("/courses")
async def courses_page(request: Request, user=Depends(login_required), db: Session = Depends(get_db)):
    courses = db.query(Course).all()
    return templates.TemplateResponse("courses.html", {
        "request": request, 
        "courses": courses,
        "user": {"username": user},
        "title": "课程管理"
    })

@app.post("/api/courses")
async def add_course(
    name: str = Form(...),
    code: str = Form(...),
    teacher: str = Form(...),
    schedule: str = Form(...),
    db: Session = Depends(get_db),
    user=Depends(login_required)
):
    try:
        course = Course(name=name, code=code, teacher=teacher, schedule_time=schedule)
        db.add(course)
        db.commit()
        return RedirectResponse(url="/courses", status_code=303)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/delete_course")
async def delete_course(course_id: int = Form(...), db: Session = Depends(get_db), user=Depends(login_required)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if course:
        db.delete(course)
        db.commit()
    return RedirectResponse(url="/courses", status_code=303)

@app.get("/analysis")
async def analysis_page(request: Request, user=Depends(login_required), db: Session = Depends(get_db)):
    # 1. Total Records
    total_records = db.query(Attendance).count()
    
    # 2. Avg Rate (Mock or Simple Calc)
    # Let's use: (Total Attendance / (Total Students * 30 days)) * 100 ?
    # Or just simpler: Total Unique Students Attended Today / Total Students
    total_students = db.query(Student).count() or 1
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_att = db.query(Attendance.student_id).filter(Attendance.timestamp >= today_start).distinct().count()
    avg_rate = int((today_att / total_students) * 100)
    
    # 3. College Stats
    colleges_stats = db.query(Student.college, func.count(Attendance.id)).join(Attendance).group_by(Student.college).all()
    col_names = [c[0] if c[0] else "Unknown" for c in colleges_stats]
    col_counts = [c[1] for c in colleges_stats]
    
    # 4. Low Attendance Warning
    # Find students with less than X attendances in last 30 days
    # Simplified: Get all students and their attendance count
    results = db.query(Student, func.count(Attendance.id)).outerjoin(Attendance).group_by(Student.id).all()
    low_att_list = []
    for s, count in results:
        if count < 5: # Threshold
            low_att_list.append({"name": s.name, "student_id": s.student_id or "-", "count": count})
            
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "user": {"username": user},
        "stats": {"avg_rate": avg_rate, "total_records": total_records},
        "chart_data": {"colleges": col_names, "counts": col_counts},
        "low_attendance": low_att_list,
        "title": "统计分析"
    })

@app.get("/api/export_attendance")
async def export_attendance(user=Depends(login_required), db: Session = Depends(get_db)):
    query = db.query(Attendance).join(Student).order_by(Attendance.timestamp.desc())
    records = query.all()
    
    data = []
    for r in records:
        data.append({
            "姓名": r.student.name,
            "学号": r.student.student_id,
            "学院": r.student.college,
            "班级": r.student.class_name,
            "打卡时间": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "状态": r.status
        })
        
    df = pd.DataFrame(data)
    
    stream = io.BytesIO()
    df.to_excel(stream, index=False)
    stream.seek(0)
    
    response = StreamingResponse(stream, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = "attachment; filename=attendance_report.xlsx"
    return response

@app.get("/settings")
async def settings_page(request: Request, user=Depends(login_required)):
    # Scan for available models
    det_weights_dir = "models/weights/detection"
    rec_weights_dir = "models/weights/recognition"
    
    det_models = []
    if os.path.exists(det_weights_dir):
        # Scan .pt files
        files = glob.glob(os.path.join(det_weights_dir, "*.pt"))
        det_models = [os.path.basename(f) for f in files]
        
    rec_models = []
    if os.path.exists(rec_weights_dir):
        # Scan directories (AdaFace, ArcFace, etc)
        subdirs = [d for d in os.listdir(rec_weights_dir) if os.path.isdir(os.path.join(rec_weights_dir, d))]
        for d in subdirs:
            # Check if .pth exists inside
            pth_files = glob.glob(os.path.join(rec_weights_dir, d, "*.pth"))
            for pth in pth_files:
                # e.g. AdaFace/best.pth
                rec_models.append(f"{d}/{os.path.basename(pth)}")
                
    # Read current config
    current_config = {}
    if os.path.exists("config/config.yaml"):
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            current_config = yaml.safe_load(f)
            
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": {"username": user},
        "det_models": det_models,
        "rec_models": rec_models,
        "config": current_config,
        "title": "系统设置"
    })

@app.post("/api/settings")
async def update_settings(
    det_model: str = Form(...),
    rec_model: str = Form(...),
    similarity_threshold: float = Form(...),
    user=Depends(login_required)
):
    try:
        config_path = "config/config.yaml"
        if not os.path.exists(config_path):
             return {"status": "error", "message": "配置文件不存在"}
             
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        # Update Config
        config['detector']['model_path'] = f"models/weights/detection/{det_model}"
        config['recognition']['weights_path'] = f"models/weights/recognition/{rec_model}"
        config['recognition']['similarity_threshold'] = similarity_threshold
        
        # Save Config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True)
            
        # Reload Service
        init_service()
        
        return RedirectResponse(url="/settings?success=1", status_code=303)
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ... (Keep existing websocket and other APIs) ...


# --- APIs ---

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    # Local session set for deduplication within a short time window
    # In a real app, this should be more robust
    recent_attendance = {} # {name: last_time}
    
    try:
        while True:
            data = await websocket.receive_text()
            if "," in data:
                header, encoded = data.split(",", 1)
            else:
                encoded = data
                
            image_data = base64.b64decode(encoded)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            if service:
                results = service.recognize_frame(frame)
            else:
                results = []
            
            # Update Attendance
            current_faces = []
            attendance_count = 0
            
            for res in results:
                name = res['name']
                if name != "Unknown":
                    # Record Attendance to DB
                    now = datetime.now()
                    if name not in recent_attendance or (now - recent_attendance[name]).seconds > 60:
                        # Find student
                        # We need to handle DB session inside WS loop carefully or use dependency
                        # Since Depends doesn't work well inside WS loop for every frame, we create a new session or reuse logic
                        # Simplified: Query by name (inefficient but works for small scale)
                        try:
                            # Re-get DB session as Depends closes it? No, Depends(get_db) works for the connection handling
                            # But inside a loop, it's better to manage transaction scope
                            stu = db.query(Student).filter(Student.name == name).first()
                            if stu:
                                att = Attendance(student_id=stu.id, timestamp=now, status="已签到")
                                db.add(att)
                                db.commit()
                                recent_attendance[name] = now
                        except Exception as e:
                            print(f"Attendance DB Error: {e}")
                            db.rollback()
                
                current_faces.append({
                    "box": res['box'],
                    "name": name,
                    "score": res['score']
                })
            
            # Get total distinct attendance count for today
            # Simplified: just return length of recent_attendance cache
            attendance_count = len(recent_attendance)
            
            await websocket.send_json({
                "faces": current_faces,
                "attendance_count": attendance_count
            })
            
    except Exception as e:
        pass
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.post("/api/register")
async def register_face(
    name: str = Form(...), 
    student_id: str = Form(None),
    college: str = Form(None),
    gender: str = Form(None),
    class_name: str = Form(None),
    file: UploadFile = File(...)
):
    if not service:
        return {"status": "error", "message": "服务未初始化"}
        
    # Ensure data/faces exists
    faces_dir = "data/faces"
    os.makedirs(faces_dir, exist_ok=True)
    
    # Save permanent file
    ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(faces_dir, f"{name}{ext}")
    
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Register in Service (In-Memory) & DB
        # Note: service.register_person only takes basic args, we need to handle DB update here or modify service
        # Let's modify the DB logic directly here for full fields support, 
        # OR update service.register_person to accept **kwargs.
        # For simplicity, let's update service.register_person signature in next step.
        # But wait, service.register_person calls DB inside. Let's pass a dict or modify it.
        
        # Actually, let's do it cleanly:
        # 1. Extract features (Service)
        feats = service.process_image(file_path)
        if not feats:
            if os.path.exists(file_path): os.remove(file_path)
            return {"status": "error", "message": "未检测到人脸"}
            
        # 2. Update In-Memory Cache
        service.known_faces[name] = feats[0]
        
        # 3. Save to DB
        db = SessionLocal()
        try:
            student = db.query(Student).filter(Student.name == name).first()
            if not student:
                student = Student(
                    name=name, 
                    face_image_path=file_path, 
                    student_id=student_id,
                    college=college,
                    gender=gender,
                    class_name=class_name
                )
                db.add(student)
            else:
                student.face_image_path = file_path
                if student_id: student.student_id = student_id
                if college: student.college = college
                if gender: student.gender = gender
                if class_name: student.class_name = class_name
            
            db.commit()
        finally:
            db.close()
        
        return RedirectResponse(url="/students", status_code=303)
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/delete_student")
async def delete_student(student_id: int = Form(...), db: Session = Depends(get_db)):
    try:
        stu = db.query(Student).filter(Student.id == student_id).first()
        if stu:
            # Delete file
            if stu.face_image_path and os.path.exists(stu.face_image_path):
                os.remove(stu.face_image_path)
            
            # Remove from DB
            db.delete(stu)
            db.commit()
            
            # Remove from memory cache
            if service and stu.name in service.known_faces:
                del service.known_faces[stu.name]
                
        return RedirectResponse(url="/students", status_code=303)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/import_students")
async def import_students(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        df = pd.read_excel(contents)
        
        # Check columns
        required_columns = ['姓名', '学号', '性别', '班级', '学院']
        if not all(col in df.columns for col in required_columns):
            return {"status": "error", "message": f"Excel必须包含列: {','.join(required_columns)}"}
            
        count = 0
        for _, row in df.iterrows():
            name = str(row['姓名']).strip()
            student_id = str(row['学号']).strip()
            
            # Skip if exists (by student_id)
            if db.query(Student).filter(Student.student_id == student_id).first():
                continue
                
            new_student = Student(
                name=name,
                student_id=student_id,
                gender=str(row['性别']).strip(),
                class_name=str(row['班级']).strip(),
                college=str(row['学院']).strip(),
                face_image_path=None # No face initially
            )
            db.add(new_student)
            count += 1
            
        db.commit()
        return RedirectResponse(url="/students", status_code=303)
    except Exception as e:
        return {"status": "error", "message": f"导入失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
