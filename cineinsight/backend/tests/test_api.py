"""Basic API tests for CineInsight backend."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.db.session import init_db


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_search_requires_query():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/movies/search")
    assert response.status_code == 422


@pytest.mark.anyio
async def test_analysis_404_for_unknown():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/movies/999999/analysis")
    assert response.status_code == 404
