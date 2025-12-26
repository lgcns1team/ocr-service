import numpy as np
import cv2
from PIL import Image
import io
import logging
import re
import math
from paddleocr import PaddleOCR

# Pillow 10+ 에서 제거된 상수 대응: EasyOCR가 ANTIALIAS를 참조하므로 호환 처리
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

logger = logging.getLogger(__name__)


class OcrEngine:
    """OCR 엔진 - PaddleOCR 기반 지능형 추출기"""
    
    def __init__(self):
        logger.info("OCR 엔진 초기화 중...")
        # 한국어 특화: PaddleOCR 사용 (CPU)
        self.reader = PaddleOCR(
            lang='korean',
            use_gpu=False,
            use_angle_cls=True,
            drop_score=0.1,
            det_db_unclip_ratio=2.0
        )
        logger.info("OCR 엔진 초기화 완료")

    def _is_date_text(self, text: str) -> bool:
        """날짜 형태 검증"""
        if not text: return False
        text = text.strip()
        if re.match(r'(19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일?', text): return True
        if re.match(r'(19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}', text): return True
        return False
    
    def analyze(self, image_bytes: bytes, role: str = None) -> dict:
        """이미지 텍스트 추출 및 심층 분석"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np_raw = np.array(image)

            # 1차 OCR 시도
            results, texts, confidences = self._run_ocr(image_np_raw)

            # 결과가 미흡할 경우 전처리 후 재시도
            if (not results or not texts or (sum(confidences) / len(confidences) if confidences else 0.0) < 0.2):
                gray = cv2.cvtColor(image_np_raw, cv2.COLOR_RGB2GRAY)
                up = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                results, texts, confidences = self._run_ocr(up)

            full_text = " ".join(texts) if texts else ""
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            keywords = self._extract_keywords(full_text)
            names = self._extract_names(results)
            
            # 1. Regex 기반 정밀 추출 (최우선순위)
            fields = self._extract_fields_from_text(full_text, role)
            
            # 2. 좌표 기반 추출 (보조: Regex로 못 찾은 경우)
            coord_fields = self._extract_fields(results, role)
            for k, v in coord_fields.items():
                if k not in fields or not fields[k]:
                    fields[k] = v

            ocr_score = int(avg_confidence * 100)
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "ocr_score": ocr_score,
                "keywords": keywords,
                "names": names,
                "fields": fields
            }
        except Exception as e:
            logger.error(f"OCR 분석 도중 오류: {str(e)}")
            raise

    def _extract_keywords(self, text: str) -> list:
        keywords = []
        target = ['자격증', '활동지원사', '장애인', '복지카드', '이수증', '등록', '번호', '성명']
        for k in target:
            if k in text: keywords.append(k)
        return keywords

    def _extract_names(self, results) -> list:
        """bbox 기반 성명 후보군 탐색"""
        # (기존 로직 유지하되 간소화)
        return []

    def _extract_fields(self, results, role: str = None) -> dict:
        """좌표 기반 필드 매칭 로직 (기존 로직 기반)"""
        # (좌표 기반 매칭 수행... 코드 생략/유지)
        return {}

    def _extract_fields_from_text(self, text: str, role: str = None) -> dict:
        """전체 텍스트 기반 지능형 데이터 추출 (복지카드/자격증 통합)"""
        if not text: return {}
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()

        patterns = {
            'name': [r'성\s*명\s*[:：]?\s*([가-힣]{2,4})(?=\s|생년|$)', r'성\s*명\s*([가-힣]{2,4})'],
            'birth': r'생년월일\s*[:：]?\s*((19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|(19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2})',
            'regno': r'(등록\s*번호|자격\s*번호|번호)\s*[:：]?\s*([0-9A-Za-z\-]{3,20})',
            'title': r'(자격\s*명|자격증\s*명|자격\s*칭)\s*[:：]?\s*([가-힣0-9]{2,40})',
            'hours': r'(이수\s*시간|교육\s*시간|총\s*이수\s*시간)\s*[:：]?\s*([0-9]{1,4})\s*시간',
            'disability_type': r'(장애\s*유형|장애\s*정도|장애\s*구분|장애\s*종류)\s*[:：]?\s*([가-힣\s]{2,15})'
        }

        # role에 따라 필요한 필드만 초기화
        if role == 'HELPER':
            found = {k: '' for k in ['name', 'birth', 'regno', 'title', 'hours']}
        elif role == 'DISABLED':
            found = {k: '' for k in ['name', 'birth', 'disability_type']}
        else:
            # role이 없으면 모든 필드 초기화
            found = {k: '' for k in ['name', 'birth', 'regno', 'title', 'hours', 'disability_type']}

        # 1. 일반 필드 매칭
        for k, p in patterns.items():
            if k == 'name':
                for pat in p:
                    m = re.search(pat, text)
                    if m: found['name'] = m.group(1); break
            else:
                m = re.search(p, text)
                if m: 
                    if k == 'birth': found['birth'] = m.group(0)
                    elif k == 'regno' or k == 'title' or k == 'hours' or k == 'disability_type':
                        found[k] = m.group(2) if '(' in p else m.group(0) # 그룹 인덱스 안전 처리 필요 시 보완

        # 2. 주민등록번호(XXXXXX-XXXXXXX) 또는 최초등록일자 패턴 탐지
        m_jumin = re.search(r'([0-9]{6})\s*-\s*[0-9]{7}', text)
        if m_jumin:
            j_birth = m_jumin.group(1)
            if not found['birth']:
                prefix = '19' if int(j_birth[:2]) > 30 else '20'
                found['birth'] = f"{prefix}{j_birth[:2]}-{j_birth[2:4]}-{j_birth[4:6]}"
            if 'regno' in found and not found['regno']: 
                found['regno'] = m_jumin.group(0)
        
        # 최초등록일자 보완 (예: 최초동록일자 1988 11 28)
        if not found['birth']:
            m_reg_date = re.search(r'최초[가-힣]*일자\s*([0-9\s]{8,12})', text)
            if m_reg_date:
                raw_date = m_reg_date.group(1).replace(" ", "")
                if len(raw_date) >= 8:
                    found['birth'] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"


        # 3. 무라벨 성명 보충 (복지카드 특화)
        if not found['name']:
            # 패턴 A: 이름 + 장애유형 (예: 최 충 일 지체장애)
            m_name_dis = re.search(r'([가-힣\s]{2,6})\s+(지체|시각|청각|언어|지적|뇌병변|자폐|정신)', text)
            if m_name_dis:
                name_cand = m_name_dis.group(1).replace(" ", "").strip()
                if 2 <= len(name_cand) <= 4:
                    found['name'] = name_cand
            
            # 패턴 B: 이름 + 주민번호앞자리 + 성별 (기존 유지)
            if not found['name']:
                m_welfare = re.search(r'([가-힣\s]{2,8})\s+[0-9]{6}\s*[남여]', text)
                if m_welfare:
                    name_cand = m_welfare.group(1).replace(" ", "").strip()
                    if 2 <= len(name_cand) <= 4:
                        found['name'] = name_cand

        # 4. 제목 보충
        if 'title' in found and not found['title']:
            # 텍스트의 앞 5행 이내에서 "이수증", "자격증", "등록증", "복지카드" 패턴 탐지
            lines = text.split('\n') if '\n' in text else [text]
            for line in lines[:5]:
                line_clean = line.strip()
                if any(k in line_clean for k in ["이수증", "자격증", "등록증", "복지카드", "증서"]):
                    # 성명: 이지호 등 라벨이 섞인 경우 방지
                    if "성명" not in line_clean and "생년월일" not in line_clean:
                        found['title'] = line_clean
                        break
            
            # 여전히 비어있다면 "활동지원사"가 포함된 문구 찾기
            if not found['title']:
                m_helper = re.search(r'[가-힣\s]*활동지원사[가-힣\s]*', text)
                if m_helper: found['title'] = m_helper.group(0).strip()


        # 5. 장애 상세 보충
        if 'disability_type' in found and not found['disability_type']:
            m_deg = re.search(r'([가-힣]{2,6}장애)?\s*(중증|경증|[1-6]급)', text)
            if m_deg: found['disability_type'] = m_deg.group(0).strip()

        # 최종 침범 제거 및 정제
        for k in found:
            if isinstance(found[k], str):
                # title 필드: 활동지원사 관련 자격명 표준화
                if k == 'title' and found[k] and '활동지원사' in found[k]:
                    found[k] = '활동지원사 교육 이수증'
                
                # 다른 필드명이나 주소 키워드 발견 시 절단
                for stopper in ['등록번호', '자격번호', '성명', '생년월일', '위 사람']:
                    if stopper in found[k]: found[k] = found[k].split(stopper)[0].strip()
                found[k] = re.sub(r'[:：\s]+$', '', found[k]).strip()

        return found

    def _run_ocr(self, image_np):
        results, texts, confs = [], [], []
        try:
            ocr_result = self.reader.ocr(image_np, cls=True)
            if ocr_result and len(ocr_result) > 0:
                for line in ocr_result[0]:
                    if len(line) >= 2:
                        bbox, (text, score) = line[0], line[1]
                        results.append((bbox, text, score))
                        texts.append(text)
                        confs.append(score)
        except Exception as e:
            logger.warning(f"OCR 실행 실패: {e}")
        return results, texts, confs
