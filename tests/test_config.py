"""Конфигурация и исключения."""
import pytest

from backend.core.config import Settings
from backend.core.exceptions import CommitteeException, WeakPasswordError


def test_dev_settings_are_valid():
    settings = Settings(DEBUG=True, _env_file=None)
    assert settings.JWT_ALGORITHM == "HS256"
    assert settings.OLLAMA_SEED == 42


def test_production_requires_strong_secret():
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(DEBUG=False, JWT_SECRET="short", _env_file=None)


def test_production_rejects_dev_default_secret():
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(DEBUG=False, _env_file=None)  # дефолтный dev-секрет запрещён в prod


def test_production_accepts_strong_secret():
    settings = Settings(DEBUG=False, JWT_SECRET="x" * 40, _env_file=None)
    assert settings.JWT_SECRET == "x" * 40


def test_exception_hierarchy():
    assert issubclass(WeakPasswordError, CommitteeException)
    exc = CommitteeException("msg", details={"a": 1})
    assert exc.message == "msg" and exc.details == {"a": 1}
