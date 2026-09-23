"""Testes para configuração do backend."""
import pytest


def test_settings_defaults():
    """Verifica que Settings carrega com valores padrão."""
    from backend.config import Settings

    s = Settings()
    assert s.app_name == "Sinaliza API"
    assert s.version == "1.0.0"
    assert s.confidence_threshold == pytest.approx(0.3)
    assert s.top_k == 5
    assert s.target_frames == 48
    assert s.device in ("cpu", "cuda", "mps", "auto")


def test_settings_env_override(monkeypatch):
    """Verifica que variáveis de ambiente sobrescrevem."""
    monkeypatch.setenv("SINALIZA_DEBUG", "true")
    monkeypatch.setenv("SINALIZA_DEVICE", "cuda")
    from backend.config import Settings

    s = Settings()
    assert s.debug is True
    assert s.device == "cuda"


def test_cors_origins_is_list():
    from backend.config import Settings

    s = Settings()
    assert isinstance(s.cors_origins, list)
    assert len(s.cors_origins) > 0
