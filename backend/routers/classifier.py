"""HTTP endpoints for the percussion classifier."""
import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from backend.services.classifier import ClassifierService
from backend.schemas.classifier import ErrorResponse, Prediction, Health
from backend.utils.errors import ErrorCode, api_error
from backend.utils.validation import validate_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["classifier"])


def get_classifier(request: Request) -> ClassifierService:
    """The service loaded once at startup (see ``server.lifespan``)."""
    return request.app.state.classifier


@router.get("/health", response_model=Health)
def health(classifier: ClassifierService = Depends(get_classifier)):
    return Health(status="ok", models=classifier.name, max_duration=classifier.max_duration)


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# CPU-heavy decoding and feature extraction do not block other requests.
@router.post(
    "/predict",
    response_model=Prediction,
    responses={
        413: {"model": ErrorResponse, "description": "FILE_TOO_LARGE"},
        415: {"model": ErrorResponse, "description": "UNSUPPORTED_FORMAT"},
        422: {"model": ErrorResponse, "description": "CORRUPT_FILE, DURATION_TOO_SHORT, DURATION_TOO_LONG, "
                                                     "SILENT_AUDIO, UNPROCESSABLE_AUDIO or INVALID_REQUEST"},
        500: {"model": ErrorResponse, "description": "INTERNAL_ERROR"},
    },
)
def predict(
        file: UploadFile = File(..., description="A .wav or .mp3 recording of one stroke"),
        classifier: ClassifierService = Depends(get_classifier),
):
    try:
        signal = validate_file(file)
        result = classifier.predict(signal)
    except HTTPException:
        raise
    except ValueError as e:
        raise api_error(422, ErrorCode.UNPROCESSABLE_AUDIO, str(e))
    except Exception:
        # Raised as an HTTPException (not left unhandled) so the response still
        # passes through CORSMiddleware; otherwise the browser reports a CORS
        # error and the frontend shows "cannot connect" instead of this error.
        logger.exception("Prediction failed for %r", file.filename)
        raise api_error(500, ErrorCode.INTERNAL_ERROR, "Prediction failed")
    return Prediction(filename=file.filename or "", **result)
