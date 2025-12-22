#!/bin/bash
# OCR 서비스 실행 스크립트 (Linux/Mac)

echo "OCR 서비스 실행 중..."

# 환경 변수 확인
if [ -z "$OCR_SERVICE_PORT" ]; then
    export OCR_SERVICE_PORT=8086
fi

# 가상 환경 확인 및 생성
if [ ! -d "venv" ]; then
    echo "가상 환경이 없습니다. 생성 중..."
    python3 -m venv venv
fi

# 가상 환경 활성화
source venv/bin/activate

# 의존성 설치 확인
if [ ! -f "venv/.installed" ]; then
    echo "의존성 설치 중..."
    pip install -r requirements.txt
    touch venv/.installed
fi

# 서비스 실행
echo "OCR 서비스 시작: http://localhost:$OCR_SERVICE_PORT"
uvicorn app.main:app --host 0.0.0.0 --port $OCR_SERVICE_PORT

