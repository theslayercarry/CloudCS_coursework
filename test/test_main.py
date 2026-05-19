# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"final_score": 82.417}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    from main import app
    return TestClient(app)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "Age": 17,
        "Gender": "Male",
        "Hours_Studied": 18.5,
        "Attendance": 92.0,
        "Sleep_Hours": 7.5,
        "Stress_Level": 4,
        "Screen_Time": 3.0,
        "Previous_GPA": 4.3,
        "Tutoring_Sessions_Per_Week": 2,
        "Exam_Anxiety_Score": 5,
        "Internet_Quality": "Good",
        "Family_Income_Level": "Middle",
        "Study_Method": "Group",
        "Part_Time_Job": "No"
    }


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=valid_payload
    )
    assert response.status_code == 200
    assert "final_score" in response.json()


def test_token_not_correctness(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer wrongtoken"},
        json=valid_payload
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        json=valid_payload
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


def test_inference(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=valid_payload
    )
    assert response.status_code == 200
    assert response.json()["final_score"] == 82.417