"""Создать администратора: python scripts/create_admin.py <email> <username> <password>.

Идемпотентен: существующий админ не дублируется.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select

from backend.core.constants import ROLE_ADMIN
from backend.core.exceptions import WeakPasswordError
from backend.db.engine import get_session_factory, init_db
from backend.db.models import User
from backend.services.auth import hash_password, validate_password


async def create_admin(email: str, username: str, password: str) -> None:
    validate_password(password)
    await init_db()
    async with get_session_factory()() as session:
        existing = (
            await session.execute(select(User).where(User.email == email.lower()))
        ).scalar_one_or_none()
        if existing is not None:
            print(f"Пользователь {email} уже существует (роль: {existing.role})")
            return
        admin = User(
            email=email.lower(),
            username=username,
            password_hash=hash_password(password),
            role=ROLE_ADMIN,
        )
        session.add(admin)
        await session.commit()
        print(f"Админ создан: {email}")


def main() -> None:
    if len(sys.argv) != 4:
        print("Использование: python scripts/create_admin.py <email> <username> <password>")
        sys.exit(1)
    try:
        asyncio.run(create_admin(sys.argv[1], sys.argv[2], sys.argv[3]))
    except WeakPasswordError as exc:
        print(f"Ошибка: {exc.message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
