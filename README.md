# OCR Service

문서 위변조 검증을 위한 OCR 서비스

## 실행 방법

### 방법 1: Gradle Task 사용 (권장)

```bash
# 의존성 설치
./gradlew :ocr-service:installPythonDeps

# 서비스 실행
./gradlew :ocr-service:runOcrService
```

### 방법 2: 실행 스크립트 사용

**Windows:**
```bash
ocr-service\run.bat
```

**Linux/Mac:**
```bash
chmod +x ocr-service/run.sh
ocr-service/run.sh
```

### 방법 3: 직접 실행

```bash
cd ocr-service

# 의존성 설치
pip install -r requirements.txt

# 서비스 실행 (환경 변수 사용)
uvicorn app.main:app --host 0.0.0.0 --port ${OCR_SERVICE_PORT:-8086}

# 또는 직접 실행
python -m app.main
```

## 환경 변수

- `OCR_SERVICE_PORT`: OCR 서비스 포트 (기본값: 8086)



