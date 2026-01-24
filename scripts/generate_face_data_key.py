import os

from cryptography.fernet import Fernet


def main():
    os.makedirs("secrets", exist_ok=True)
    path = os.path.join("secrets", "face_data.key")
    if os.path.exists(path):
        print("Key already exists:", path)
        return
    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    print("Generated:", path)


if __name__ == "__main__":
    main()

