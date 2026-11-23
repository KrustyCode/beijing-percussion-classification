"""FastAPI app for the percussion classifier.

Run from the project root (so `config`, `model`, ... and the pickled models
can be imported):

    uvicorn backend.server:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from backend.routers import classifier
from backend.services.classifier import ClassifierService
from backend.utils.errors import ErrorCode, error_detail

# The saved model that answers every request: {name used by the router: models/<file>.pkl}.
# The key must match BEST_MODEL in routers/classifier.py.
MODEL_FILE = "all_svm"

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

ALLOWED_METHODS = [
    "GET",
    "POST",          # /api/predict uploads a file
]

ALLOWED_HEADERS = [
    "*"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the models into memory ...
    app.state.classifier = ClassifierService.from_file(MODEL_FILE)
    yield
    # ... and once at shutdown
    app.state.classifier = None


app = FastAPI(title="Percussion classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS
)

app.include_router(classifier.router)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    """FastAPI's own 422 (e.g. no `file` field) in the same shape as the others."""
    fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": error_detail(ErrorCode.INVALID_REQUEST, f"Invalid request: {fields}")},
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")
