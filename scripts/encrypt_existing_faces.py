import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.crypto_manager import encrypt_bytes
from core.database import SessionLocal
from core.models import Student


def main():
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        changed = 0
        for s in students:
            if not s.face_image_path:
                continue
            path = str(s.face_image_path)
            if path.endswith(".enc"):
                continue
            if not os.path.exists(path):
                continue

            with open(path, "rb") as f:
                raw = f.read()

            new_path = os.path.splitext(path)[0] + ".enc"
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            with open(new_path, "wb") as f:
                f.write(encrypt_bytes(raw))

            try:
                os.remove(path)
            except Exception:
                pass

            s.face_image_path = new_path
            db.add(s)
            changed += 1

        db.commit()
        print("encrypted_files:", changed)
    finally:
        db.close()


if __name__ == "__main__":
    main()
