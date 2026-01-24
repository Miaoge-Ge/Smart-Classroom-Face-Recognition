from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Float
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

class UserAccount(Base):
    __tablename__ = "user_accounts"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, index=True, nullable=False)  # admin | teacher | student
    full_name = Column(String, nullable=True)
    student_id = Column(Integer, ForeignKey("students.student_id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    student = relationship("Student")

class Course(Base):
    __tablename__ = "courses"
    course_id = Column(Integer, primary_key=True, index=True)
    course_no = Column(String, unique=True, index=True, nullable=False)
    course_name = Column(String, index=True, nullable=False)
    teacher = Column(String, nullable=True)
    schedule = Column(String, nullable=True)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    attendances = relationship("Attendance", back_populates="course")

class Student(Base):
    __tablename__ = "students"

    student_id = Column(Integer, primary_key=True, index=True)
    student_no = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, index=True, nullable=False)
    class_name = Column("class", String, nullable=True)
    college = Column(String, nullable=True) # 学院
    gender = Column(String, nullable=True) # 性别
    
    face_image_path = Column(String) # Path to the original registered image
    face_embedding_enc = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

    attendances = relationship("Attendance", back_populates="student", cascade="all, delete-orphan", passive_deletes=True)

class Attendance(Base):
    __tablename__ = "attendances"

    record_id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.student_id", ondelete="CASCADE"), index=True, nullable=False)
    course_id = Column(Integer, ForeignKey("courses.course_id", ondelete="CASCADE"), index=True, nullable=False)
    check_time = Column(DateTime, default=datetime.now, index=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, default="Present") # Present, Late, etc.
    created_at = Column(DateTime, default=datetime.now, index=True)
    
    student = relationship("Student", back_populates="attendances")
    course = relationship("Course", back_populates="attendances")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_username = Column(String, index=True, nullable=True)
    actor_role = Column(String, index=True, nullable=True)
    action = Column(String, index=True, nullable=False)
    resource = Column(String, index=True, nullable=True)
    status = Column(String, nullable=True)
    ip = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, index=True)
