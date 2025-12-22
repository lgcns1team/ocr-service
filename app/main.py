from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.ocr.engine import OcrEngine
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Service", version="1.0.0")
ocr_engine = OcrEngine()

@app.post("/api/ocr/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    이미지 파일에서 텍스트를 추출하고 분석
    
    Args:
        file: 분석할 이미지 파일
        
    Returns:
        OCR 분석 결과 (extractedText, confidence, ocrScore, keywords)
    """
    try:
        logger.info(f"OCR 분석 요청: {file.filename}")
        
        image_bytes = await file.read()
        result = ocr_engine.analyze(image_bytes)
        
        return {
            "extractedText": result["text"],
            "confidence": result["confidence"],
            "ocrScore": result["ocr_score"],
            "keywords": result["keywords"],
            "names": result["names"],
            "fields": result.get("fields", {})
        }
    except Exception as e:
        logger.error(f"OCR 분석 실패: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "OCR 처리에 실패했습니다.", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("OCR_SERVICE_PORT", "8086"))
    uvicorn.run(app, host="0.0.0.0", port=port)



