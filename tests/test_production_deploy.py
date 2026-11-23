"""CORS configurável e mídia real do sinalário no mesmo app publicado."""
import json

from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.config import Settings, settings
from backend.main import app
from backend.routers.dictionary import DICTIONARY_DIR, _load_from_folder


def test_cors_por_variavel_de_ambiente(monkeypatch):
    allowed = "https://sinaliza.vercel.app"
    monkeypatch.setenv("SINALIZA_CORS_ORIGINS", json.dumps([allowed]))
    monkeypatch.setenv("SINALIZA_CORS_ORIGIN_REGEX", r"https://sinaliza-preview-[a-z0-9-]+\.vercel\.app")
    config = Settings(_env_file=None)
    assert config.cors_origins == [allowed]
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    assert cors.kwargs["allow_origin_regex"] == settings.cors_origin_regex
    monkeypatch.setitem(cors.kwargs, "allow_origins", config.cors_origins)
    monkeypatch.setitem(cors.kwargs, "allow_origin_regex", config.cors_origin_regex)
    monkeypatch.setattr(app, "middleware_stack", None)
    client = TestClient(app)
    for origin in (allowed, "https://sinaliza-preview-abc123.vercel.app"):
        response = client.options("/api/health", headers={"Origin": origin,
                                  "Access-Control-Request-Method": "GET",
                                  "Access-Control-Request-Headers": "content-type"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert client.get("/api/dictionary?limit=1", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
    for origin in ("https://outro.vercel.app", "https://sinaliza.vercel.app.evil.example"):
        response = client.options("/api/health", headers={"Origin": origin,
                                  "Access-Control-Request-Method": "GET"})
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers


def test_sinalario_tem_midia_publicada_e_suporta_range():
    entries = _load_from_folder()
    videos = [entry for entry in entries if entry.example_video_url]
    assert videos, f"Vídeos preparados ausentes em {DICTIONARY_DIR}"
    client = TestClient(app)
    entry = videos[0]
    response = client.get(entry.example_video_url, headers={"Range": "bytes=0-31"})
    assert response.status_code == 206
    assert response.headers["content-type"].startswith("video/mp4")
    assert len(response.content) == 32
    assert response.headers["content-range"].startswith("bytes 0-31/")
    assert entry.poster_url and client.get(entry.poster_url).status_code == 200
    image = next(entry for entry in entries if entry.image_url)
    assert client.get(image.image_url).status_code == 200
