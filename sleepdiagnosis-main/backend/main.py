from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import os
import uuid
from model import load_model, predict_from_file
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "saved_model.joblib"

app = FastAPI(
    title="SleepGuard API", 
    description="Sleep apnea detection through snore analysis",
    version="1.0.0"
)

# Allow your frontend origin(s) here
origins = [
    "http://localhost:3000",
    "http://localhost:9002",  # Added for Next.js dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:9002",
    "https://sleepguard.vercel.app",
    "*"  # Allow all origins for development - remove in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def startup_event():
    global model
    logger.info("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        if model:
            logger.info("Model loaded successfully")
        else:
            logger.warning("No model file found, using fallback heuristics")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "SleepGuard API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "service": "SleepGuard API",
        "version": "1.0.0"
    }

@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    logger.info(f"Received file: {audio.filename}, content-type: {audio.content_type}, size: {audio.size if hasattr(audio, 'size') else 'unknown'}")
    
    # Enhanced file validation - prioritize MP3 and WAV for optimal spectrogram generation
    valid_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.webm', '.ogg')
    # Accept more content types for browser recordings, prioritize MP3
    valid_content_types = ('audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/', 'video/webm', 'application/octet-stream', None)
    
    file_extension = os.path.splitext(audio.filename)[1].lower() if audio.filename else ''
    content_type_valid = (audio.content_type is None or 
                         any(audio.content_type.startswith(ct) for ct in valid_content_types if ct is not None))
    
    logger.info(f"File validation: extension={file_extension}, content_type={audio.content_type}")
    
    # Log if MP3 format detected for optimal processing
    if file_extension == '.mp3' or (audio.content_type and 'mpeg' in audio.content_type):
        logger.info("MP3 format detected - optimal for spectrogram generation and sleep apnea analysis")
    
    # For browser recordings, be more lenient - allow if either extension OR content type is valid
    if file_extension and file_extension not in valid_extensions and not content_type_valid:
        logger.warning(f"Invalid file: extension={file_extension}, content-type={audio.content_type}")
        raise HTTPException(
            status_code=400, 
            detail="Invalid audio file. Please upload a supported audio file (MP3 recommended for best results)."
        )

    # Create tmp directory if it doesn't exist
    os.makedirs("tmp_uploads", exist_ok=True)
    
    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(audio.filename)[1] or ".wav"
    tmp_path = os.path.join("tmp_uploads", file_id + ext)

    try:
        # Read and validate file size
        file_content = await audio.read()
        file_size = len(file_content)
        logger.info(f"File read successfully: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        if file_size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50MB)")
        
        with open(tmp_path, "wb") as buffer:
            buffer.write(file_content)
        
        logger.info(f"File saved to {tmp_path} ({file_size} bytes), processing...")
        
        # Verify file exists and is readable
        if not os.path.exists(tmp_path):
            raise Exception("File was not saved properly")
            
        actual_size = os.path.getsize(tmp_path)
        logger.info(f"File verification: expected {file_size} bytes, actual {actual_size} bytes")
        
        # Process the audio file
        result = predict_from_file(tmp_path, model)
        
        logger.info(f"Analysis complete: {result['label']} with probability {result['probability']:.3f}")
        
        # Return formatted result
        return JSONResponse(content={
            "success": True,
            "label": result["label"],
            "probability": result["probability"],
            "confidence_score": round(result["probability"] * 100),
            "features": result.get("features", {}),
            "spectrogram": result.get("spectrogram", {}),
            "note": result.get("note", "Analysis completed"),
            "timestamp": file_id
        })
        
    except Exception as e:
        logger.error(f"Processing error for file {audio.filename}: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Provide more specific error messages
        error_message = str(e)
        if "Empty audio file" in error_message:
            status_code = 400
        elif "Could not load audio" in error_message or "librosa" in error_message:
            status_code = 422
            error_message = "Audio file format not supported or corrupted"
        elif "No such file" in error_message:
            status_code = 500
            error_message = "File processing error"
        else:
            status_code = 500
            error_message = f"Audio processing failed: {error_message}"
            
        raise HTTPException(status_code=status_code, detail=error_message)
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.info(f"Cleaned up temporary file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {tmp_path}: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
