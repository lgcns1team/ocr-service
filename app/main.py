from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.ocr.engine import OcrEngine
import os
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OCR Service", version="1.0.0")
ocr_engine = OcrEngine()

class OcrUrlRequest(BaseModel):
    fileUrl: str
    role: str = None

def process_ocr(image_bytes: bytes, role: str = None):
    """OCR 공통 처리 로직"""
    result = ocr_engine.analyze(image_bytes, role)
    
    # 디버깅용 로그
    logger.info(f"=== OCR 분석 결과 ===")
    logger.info(f"Fields: {result.get('fields', {})}")
    logger.info(f"Names: {result.get('names', [])}")
    logger.info(f"====================")
    
    return {
        "extractedText": result["text"],
        "confidence": result["confidence"],
        "ocrScore": result["ocr_score"],
        "keywords": result["keywords"],
        "names": result["names"],
        "fields": result.get("fields", {})
    }

@app.post("/api/ocr/extract")
async def extract_from_url(request: OcrUrlRequest):
    """
    S3 URL에서 이미지를 다운로드하여 텍스트 추출
    """
    try:
        logger.info(f"URL 기반 OCR 추출 요청: {request.fileUrl}, role: {request.role}")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(request.fileUrl)
            if response.status_code != 200:
                raise Exception(f"이미지 다운로드 실패 (상태 코드: {response.status_code})")
            image_bytes = response.content
            
        return process_ocr(image_bytes, request.role)
        
    except Exception as e:
        logger.error(f"URL OCR 추출 실패: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "OCR 처리에 실패했습니다.", "message": str(e)}
        )

@app.post("api/ocr/analyze")
async def analyze_document(file: UploadFile = File(...), role: str = Form(None)):
    """
    이미지 파일에서 텍스트를 추출하고 분석 (기존 방식)
    """
    try:
        logger.info(f"파일 기반 OCR 분석 요청: {file.filename}, role: {role}")
        image_bytes = await file.read()
        return process_ocr(image_bytes, role)
        
    except Exception as e:
        logger.error(f"파일 OCR 분석 실패: {str(e)}", exc_info=True)
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



