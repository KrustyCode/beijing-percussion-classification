from pydantic import BaseModel

class Prediction(BaseModel):
    label: str
    analysed_seconds: float
    processing_ms: float


class Health(BaseModel):
    status: str
    models: str
    max_duration: float | None

class ErrorDetail(BaseModel):
    code: str       
    message: str

class ErrorResponse(BaseModel):
    detail: ErrorDetail
