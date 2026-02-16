from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_login_without_admin_returns_401():
    response = client.post(
        "/api/login",
        json={"username": "unknown", "password": "Password1!"},
    )
    assert response.status_code in {401, 400}


def test_register_admin_requires_auth():
    response = client.post(
        "/api/admin/register",
        json={
            "email": "new@test.com",
            "username": "newadmin",
            "password": "Password1!",
        },
    )
    assert response.status_code in {401, 403}
