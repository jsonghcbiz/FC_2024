from pydantic import BaseModel

class BertInput(BaseModel):
    text: list[str]

class BertOutput(BaseModel):
    model_name: str
    text: list[str]
    labels: list[str]
    scores: list[str]
    prediction_time: float    # 초 데이터

