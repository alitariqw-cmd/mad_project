from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import os
from model import AlzheimerModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Alzheimer Detection API",
    description="API for detecting Alzheimer's disease from brain scan images",
    version="1.0.0"
)

# Add CORS middleware to allow requests from Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), "..", "ai_and_mad_project", "alzheimer_coatnet.onnx")
        
        # Try alternative path if first one fails
        if not os.path.exists(model_path):
            model_path = "alzheimer_coatnet.onnx"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = AlzheimerModel(model_path)
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Alzheimer Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict Alzheimer's disease from brain scan image
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        JSON with prediction results
    """
    try:
        # Validate file
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file type
        allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed types: {allowed_types}"
            )
        
        # Read file
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Run inference
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        results = model.predict(contents)
        
        return {
            "success": True,
            "prediction": results,
            "filename": file.filename,
            "message": "Prediction successful"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.get("/info")
async def model_info():
    """Get model information"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        return {
            "model_type": "ONNX CoAtNet",
            "purpose": "Alzheimer's disease detection",
            "input_shape": str(model.input_shape),
            "input_name": model.input_name,
            "output_name": model.output_name,
            "status": "Ready for inference"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
