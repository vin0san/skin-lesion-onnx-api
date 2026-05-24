import logging
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
import uvicorn

from app.inference import SkinInference, Config

# ============================================================================
# CONFIG
# ============================================================================

class APIConfig:
    """API configuration."""
    MODEL_PATH = os.getenv("MODEL_PATH", "models/skin_classifier_v1.onnx")
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
    UNCERTAINTY_THRESHOLD = 0.50
    MELANOMA_CLASS = "Melanoma"


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Skin Lesion Classifier API",
    description="EfficientNet-B3 ONNX inference server for dermoscopic lesion classification",
    version="1.0.0"
)

# Global inference engine
predictor: Optional[SkinInference] = None


@app.on_event("startup")
async def startup_event():
    """Initialize inference engine on startup."""
    global predictor
    
    try:
        if not os.path.exists(APIConfig.MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {APIConfig.MODEL_PATH}")
        
        predictor = SkinInference(APIConfig.MODEL_PATH)
        logger.info("Inference engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "EfficientNet-B3 ONNX",
        "version": "1.0.0"
    }


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/predict", tags=["inference"])
async def predict_lesion(file: UploadFile = File(...)):
    """
    Classify a skin lesion from an uploaded image.
    
    Args:
        file: Image file (JPEG or PNG)
    
    Returns:
        JSON response with prediction, confidence, and clinical notes
    
    Raises:
        400: Invalid file format or size
        422: Image cannot be processed
        500: Server error
    """
    
    # 1. Validate file type
    if file.content_type not in APIConfig.ALLOWED_CONTENT_TYPES:
        logger.warning(f"Invalid content type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: JPEG, PNG. Got: {file.content_type}"
        )
    
    # 2. Read and validate file size
    try:
        image_bytes = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read file"
        )
    
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty"
        )
    
    if len(image_bytes) > APIConfig.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large (max {APIConfig.MAX_FILE_SIZE_MB}MB)"
        )
    
    # 3. Run inference
    try:
        if predictor is None:
            logger.error("Predictor not initialized")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model not initialized"
            )
        
        result = predictor.predict_with_metadata(image_bytes)
        
    except ValueError as e:
        # Preprocessing or image decoding error
        logger.warning(f"Preprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot process image: {str(e)}"
        )
    except Exception as e:
        # Unexpected error
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during inference"
        )
    
    # 4. Format response
    top_pred = result["prediction"]
    top_conf = result["confidence"]
    is_uncertain = result["is_uncertain"]
    
    # Clinical decision support
    needs_dermatologist = (
        is_uncertain or 
        top_pred == APIConfig.MELANOMA_CLASS
    )
    
    return {
        "filename": file.filename,
        "prediction": top_pred,
        "confidence": round(top_conf, 4),
        "is_uncertain": is_uncertain,
        "requires_dermatologist_review": needs_dermatologist,
        "clinical_note": (
            "High uncertainty or melanoma suspicion. Recommend dermatologist consultation."
            if needs_dermatologist
            else "Monitor for changes over time."
        ),
        "all_probabilities": {
            name: round(prob, 4)
            for name, prob in result["all_predictions"]
        }
    }


@app.post("/predict_batch", tags=["inference"])
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Classify multiple skin lesions from uploaded images.
    
    Args:
        files: List of image files (JPEG or PNG)
    
    Returns:
        JSON response with batch predictions
    """
    
    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    if len(files) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 files per batch"
        )
    
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            image_bytes = await file.read()
            
            if len(image_bytes) > APIConfig.MAX_FILE_SIZE_BYTES:
                errors.append({
                    "index": idx,
                    "filename": file.filename,
                    "error": "File too large"
                })
                continue
            
            result = predictor.predict_with_metadata(image_bytes)
            
            results.append({
                "index": idx,
                "filename": file.filename,
                "prediction": result["prediction"],
                "confidence": round(result["confidence"], 4),
                "is_uncertain": result["is_uncertain"],
            })
        
        except Exception as e:
            errors.append({
                "index": idx,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }


# ============================================================================
# ROOT
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """API information."""
    return {
        "name": "Skin Lesion Classifier API",
        "version": "1.0.0",
        "model": "EfficientNet-B3 (ONNX)",
        "classes": Config.CLASSES,
        "endpoints": {
            "health": "GET /health",
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict_batch",
            "docs": "/docs"
        }
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )