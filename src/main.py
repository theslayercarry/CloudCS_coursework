# -*- coding: utf-8 -*-
import os

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from model_utils import load_model, make_inference


class Instance(BaseModel):
    Age: int
    Gender: str
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Stress_Level: int
    Screen_Time: float
    Previous_GPA: float
    Tutoring_Sessions_Per_Week: int
    Exam_Anxiety_Score: int
    Internet_Quality: str
    Family_Income_Level: str
    Study_Method: str
    Part_Time_Job: str


app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")

model_path: str | None = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def is_token_correct(token: str) -> bool:
    dummy_correct_token = "00000"
    return token == dummy_correct_token


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    if not await is_token_correct(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions")
async def predictions(
    instance: Instance,
    token: str = Depends(check_token)
) -> dict[str, float]:
    return make_inference(load_model(model_path), instance.dict())