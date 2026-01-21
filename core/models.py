from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class AdminUser(Base):
    __tablename__ = "admin_users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True) # e.g. "Advanced Mathematics"
    code = Column(String, unique=True, index=True) # e.g. "MATH101"
    teacher = Column(String, nullable=True)
    schedule_time = Column(String, nullable=True) # e.g. "Mon 10:00-12:00"
    created_at = Column(DateTime, default=datetime.now)

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    student_id = Column(String, unique=True, index=True, nullable=True) # 学号
    college = Column(String, nullable=True) # 学院
    gender = Column(String, nullable=True) # 性别
    class_name = Column(String, nullable=True) # 班级
    
    face_image_path = Column(String) # Path to the original registered image
    created_at = Column(DateTime, default=datetime.now)

    attendances = relationship("Attendance", back_populates="student")

class Attendance(Base):
    __tablename__ = "attendances"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String, default="Present") # Present, Late, etc.
    course_name = Column(String, nullable=True) # 课程名称
    
    student = relationship("Student", back_populates="attendances")
