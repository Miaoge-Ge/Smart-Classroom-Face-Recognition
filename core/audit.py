import json
from datetime import datetime

from core.database import SessionLocal
from core.models import AuditLog


def write_audit_log(
    *,
    actor_username: str | None,
    actor_role: str | None,
    action: str,
    resource: str | None = None,
    status: str | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
    details: dict | None = None,
) -> None:
    db = SessionLocal()
    try:
        row = AuditLog(
            actor_username=actor_username,
            actor_role=actor_role,
            action=action,
            resource=resource,
            status=status,
            ip=ip,
            user_agent=user_agent,
            details=json.dumps(details, ensure_ascii=False) if details is not None else None,
            created_at=datetime.now(),
        )
        db.add(row)
        db.commit()
    finally:
        db.close()

