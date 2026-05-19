# -*- coding: utf-8 -*-
import joblib
import pandas as pd


def make_inference(in_model, in_data: dict) -> dict[str, float]:
    prediction = in_model.predict(pd.DataFrame(in_data, index=[0]))[0]
    return {"final_score": round(float(prediction), 3)}


def load_model(path: str):
    return joblib.load(path)