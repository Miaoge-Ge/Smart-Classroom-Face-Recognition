import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from core.database import SessionLocal
from core.models import Student
from web.app import app


def main():
    client = TestClient(app)

    resp = client.post("/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
    assert resp.status_code in (302, 303)

    resp = client.get("/settings")
    assert resp.status_code == 200

    teacher_username = "teacher_demo"
    resp = client.post(
        "/api/users",
        data={"username": teacher_username, "password": "teacher123", "role": "teacher", "full_name": "Teacher Demo"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)

    student_no = "S999999"
    db = SessionLocal()
    try:
        stu = db.query(Student).filter(Student.student_no == student_no).first()
        if not stu:
            stu = Student(name="Student Demo", student_no=student_no)
            db.add(stu)
            db.commit()
    finally:
        db.close()

    student_username = "student_demo"
    resp = client.post(
        "/api/users",
        data={
            "username": student_username,
            "password": "student123",
            "role": "student",
            "full_name": "Student Demo",
            "student_no": student_no,
        },
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)

    teacher = TestClient(app)
    resp = teacher.post("/login", data={"username": teacher_username, "password": "teacher123"}, follow_redirects=False)
    assert resp.status_code in (302, 303)

    resp = teacher.get("/settings", follow_redirects=False)
    assert resp.status_code == 403

    resp = teacher.get("/history", follow_redirects=False)
    assert resp.status_code == 200

    student = TestClient(app)
    resp = student.post("/login", data={"username": student_username, "password": "student123"}, follow_redirects=False)
    assert resp.status_code in (302, 303)

    resp = student.get("/my_attendance", follow_redirects=False)
    assert resp.status_code == 200

    resp = student.get("/history", follow_redirects=False)
    assert resp.status_code == 403

    print("OK")


if __name__ == "__main__":
    main()
