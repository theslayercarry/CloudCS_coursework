# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from model_utils import make_inference, load_model
from sklearn.pipeline import Pipeline
from pickle import dumps


@pytest.fixture
def create_data() -> dict:
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


def test_make_inference(monkeypatch, create_data):
    def mock_get_predictions(_, data: pd.DataFrame):
        assert create_data == {
            key: value[0] for key, value in data.to_dict("list").items()
        }
        return [76.284]

    in_model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_get_predictions)

    result = make_inference(in_model, create_data)
    assert result == {"final_score": 76.284}


@pytest.fixture()
def filepath_and_data(tmpdir):
    p = tmpdir.mkdir("datadir").join("fakedmodel.pkl")
    example: str = "Test message!"
    p.write_binary(dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    assert filepath_and_data[1] == load_model(filepath_and_data[0])