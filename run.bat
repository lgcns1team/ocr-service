@echo off
REM OCR 서비스 실행 스크립트 (Windows)

echo OCR 서비스 실행 중...

REM 환경 변수 확인
if "%OCR_SERVICE_PORT%"=="" (
    set OCR_SERVICE_PORT=8086
)

REM 의존성 설치 확인
if not exist "venv\" (
    echo 가상 환경이 없습니다. 생성 중...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM 서비스 실행
echo OCR 서비스 시작: http://localhost:%OCR_SERVICE_PORT%
uvicorn app.main:app --host 0.0.0.0 --port %OCR_SERVICE_PORT%

