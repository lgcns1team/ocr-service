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
    """OCR 엔진 - EasyOCR 사용"""
    
    def __init__(self):
        logger.info("OCR 엔진 초기화 중...")
        # 한국어 특화: PaddleOCR 사용 (CPU)
        self.reader = PaddleOCR(
            lang='korean',
            use_gpu=False,
            use_angle_cls=True,
            drop_score=0.1
        )
        logger.info("OCR 엔진 초기화 완료")

    def _is_date_text(self, text: str) -> bool:
        """
        날짜 형태인지 검증 (YYYY-MM-DD / YYYY.MM.DD / YYYY년 MM월 DD일)
        """
        if not text:
            return False
        text = text.strip()
        # 년월일 표기
        m = re.match(r'(19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일?', text)
        if m:
            return True
        # 구분자 -, . , /
        m = re.match(r'(19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}', text)
        if m:
            year = int(text[:4])
            parts = re.split(r'[.\-/]', text)
            if len(parts) >= 3:
                try:
                    month = int(parts[1])
                    day = int(parts[2])
                    return 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100
                except Exception:
                    return False
        return False
    
    def analyze(self, image_bytes: bytes) -> dict:
        """
        이미지에서 텍스트 추출 및 분석
        
        Args:
            image_bytes: 이미지 바이트 데이터
            
        Returns:
            {
                "text": 추출된 텍스트,
                "confidence": 평균 신뢰도,
                "ocr_score": OCR 점수 (0-100),
                "keywords": 발견된 키워드 목록,
                "names": 추출된 이름 후보 목록
            }
        """
        try:
            # 이미지 로드
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_np_raw = np.array(image)

            # 1차: 원본 그대로 (PaddleOCR)
            image_np = image_np_raw
            results, texts, confidences = self._run_ocr(image_np)

            # 2차: 신뢰도/텍스트가 거의 없으면 fallback (업샘플+부드러운 전처리)
            if (not results or not texts or (sum(confidences) / len(confidences) if confidences else 0.0) < 0.2):
                try:
                    gray = cv2.cvtColor(image_np_raw, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    up = cv2.resize(blur, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
                    results_fallback, texts_fb, confs_fb = self._run_ocr(up)
                    if texts_fb:
                        results = results_fallback
                        texts = texts_fb
                        confidences = confs_fb
                except Exception as fe:
                    logger.warning(f"Fallback OCR 실패: {fe}")

            # 3차: 여전히 텍스트가 없거나 신뢰도 낮을 때 강한 대비 시도 (이진화)
            if (not results or not texts or (sum(confidences) / len(confidences) if confidences else 0.0) < 0.15):
                try:
                    gray = cv2.cvtColor(image_np_raw, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
                    up = cv2.resize(th, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
                    results_fallback2, texts_fb2, confs_fb2 = self._run_ocr(up)
                    if texts_fb2:
                        results = results_fallback2
                        texts = texts_fb2
                        confidences = confs_fb2
                except Exception as fe:
                    logger.warning(f"Fallback OCR(이진화) 실패: {fe}")

            if not results or not texts:
                return {
                    "text": "",
                    "confidence": 0.0,
                    "ocr_score": 0,
                    "keywords": [],
                    "names": [],
                    "fields": {}
                }
            
            # 텍스트 추출 및 신뢰도 계산
            # texts/confidences는 위에서 파싱 완료
            
            extracted_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 키워드/이름 추출
            keywords = self._extract_keywords(extracted_text)
            names = self._extract_names(results)
            fields = self._extract_fields(results)
            # 텍스트 기반 보완 추출을 병합 (위치 기반에서 못 잡은 필드 보충)
            text_fields = self._extract_fields_from_text(extracted_text)
            for k, v in text_fields.items():
                if not fields.get(k):
                    fields[k] = v
            # names 비어있으면 fields의 name을 names에도 반영
            if (not names) and fields.get('name'):
                names = [fields['name']]
            
            # OCR 점수 계산 (신뢰도 기반)
            ocr_score = int(avg_confidence * 100)
            
            return {
                "text": extracted_text,
                "confidence": float(avg_confidence),
                "ocr_score": ocr_score,
                "keywords": keywords,
                "names": names,
                "fields": fields
            }
            
        except Exception as e:
            logger.error(f"OCR 분석 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _extract_keywords(self, text: str) -> list:
        """
        문서 타입별 키워드 추출
        """
        keywords = []
        keyword_list = [
            '자격증', '활동지원사', '장애인', '증명서',
            '발급', '기관', '번호', '이름', '생년월일',
            '주소', '등록', '인증', '확인'
        ]
        
        for keyword in keyword_list:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords

    def _extract_names(self, results) -> list:
        """
        bbox 기반으로 '성명' 오른쪽 같은 라인에 있는 2~4자 한글을 이름 후보로 추출
        """
        if not results:
            return []

        # 결과 형식: [bbox, text, conf]
        name_candidates = []
        label_positions = []

        def get_line_key(bbox):
            ys = [p[1] for p in bbox]
            return (min(ys) + max(ys)) / 2.0

        # '성명' 라벨 위치 수집 (공백/변형 허용)
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[0], r[1], r[2]
                if conf is None:
                    continue
                if re.search(r'성\s*명', text):
                    line_y = get_line_key(bbox)
                    x_center = (bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 4.0
                    label_positions.append((line_y, x_center))

        if not label_positions:
            return []

        # 같은 라인에 있고, 라벨보다 오른쪽에 있는 2~4자 한글 토큰 중 가장 가까운 것
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[0], r[1], r[2]
                if conf is None:
                    continue
                # 2~4자 한글만
                if not re.fullmatch(r'[가-힣]{2,4}', text):
                    continue
                line_y = get_line_key(bbox)
                x_center = (bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 4.0

                for lp_y, lp_x in label_positions:
                    if abs(line_y - lp_y) < 25 and x_center > lp_x:  # 같은 라인 허용 오차 완화
                        name_candidates.append((abs(line_y - lp_y) + abs(x_center - lp_x), text))

        if not name_candidates:
            return []

        name_candidates.sort(key=lambda x: x[0])
        return [name_candidates[0][1]]

    def _extract_fields(self, results) -> dict:
        """
        라벨-값 추출 (성명, 생년월일 등)
        """
        fields = {}
        if not results:
            return fields

        def get_line_key(bbox):
            ys = [p[1] for p in bbox]
            return (min(ys) + max(ys)) / 2.0

        labels = {
            'name': r'성\s*명',
            'birth': r'생\s*년\s*월\s*일|생년월일',
            'regno': r'등록\s*번호|등록번호',
            'title': r'(자격증명|자격증\s*제목|자격명|자격증|교육과정|과정명|교육과정명)',
            'hours': r'(교육시간|이수시간|총\s*이수시간|교육\s*이수시간|교육\s*총\s*시간|총\s*시간)'
        }

        label_pos = {k: [] for k in labels}
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[0], r[1], r[2]
                if conf is None:
                    continue
                for key, pat in labels.items():
                    if re.search(pat, text):
                        line_y = get_line_key(bbox)
                        x_center = (bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 4.0
                        label_pos[key].append((line_y, x_center))

        def is_valid_value(key: str, text: str) -> bool:
            # 라벨별 값 포맷 검증
            patterns = {
                'name': r'[가-힣]{2,4}',
                'birth': r'.*',  # 아래에서 별도 검증
                'regno': r'[0-9A-Za-z\-]{3,20}',
                'title': r'[가-힣A-Za-z0-9\s\-]{2,40}',
                'hours': r'[0-9]{1,4}\s*시간'
            }
            pat = patterns.get(key, r'[가-힣0-9\-~\s]{2,20}')
            if re.fullmatch(pat, text) is None:
                return False
            if key == 'birth':
                # 날짜 형태 검증 (잘못된 번호를 생년월일로 오인하지 않도록)
                return self._is_date_text(text)
            return True

        # 값 후보: 라벨보다 오른쪽, 같은 라인
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[0], r[1], r[2]
                if conf is None:
                    continue
                line_y = get_line_key(bbox)
                x_center = (bbox[0][0] + bbox[1][0] + bbox[2][0] + bbox[3][0]) / 4.0

                for key, pos_list in label_pos.items():
                    for lp_y, lp_x in pos_list:
                        if abs(line_y - lp_y) < 25 and x_center > lp_x and is_valid_value(key, text):
                            # 가장 가까운 값만 채택
                            prev = fields.get(key)
                            dist = abs(line_y - lp_y) + abs(x_center - lp_x)
                            if prev is None or dist < prev[0]:
                                fields[key] = (dist, text)

        # dist 제거하고 텍스트만 반환
        return {k: v[1] for k, v in fields.items()}

    def _extract_fields_from_text(self, text: str) -> dict:
        """
        라벨 매칭이 실패했을 때 전체 텍스트에서 정규식으로 필드 추출
        """
        if not text:
            return {}

        # 공백/콜론 등 normalize
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()

        patterns = {
            'name': [
                r'성\s*명\s*[:：]?\s*([가-힣]{2,4})',
                r'성\s*명[^가-힣]{0,5}([가-힣]{2,4})'  # 라벨과 이름 사이에 노이즈 허용
            ],
            'birth': r'(19|20)\d{2}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|'
                     r'(19|20)\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}',
            'regno': r'등록번호\s*[:：]?\s*([0-9A-Za-z\-]{3,20})',
            'title': r'(자격증명|자격증\s*제목|자격명|자격증|교육과정|과정명)\s*[:：]?\s*([가-힣A-Za-z0-9\s\-]{3,40})',
            'hours': r'(이수시간|교육시간|총\s*이수시간|교육\s*총\s*시간)\s*[:：]?\s*([0-9]{1,4})\s*시간'
        }

        found = {}

        for pat in patterns['name']:
            m_name = re.search(pat, text)
            if m_name:
                found['name'] = m_name.group(1)
                break
        # 추가 휴리스틱: '성 명' 이후 연속 한글 2~4자 추출
        if 'name' not in found:
            m = re.search(r'성\s*명[^가-힣]{0,10}([가-힣]{2,4})', text)
            if m:
                found['name'] = m.group(1)

        m_birth = re.search(patterns['birth'], text)
        if m_birth:
            cand = m_birth.group(0).strip()
            if self._is_date_text(cand):
                found['birth'] = cand

        m_reg = re.search(patterns['regno'], text)
        if m_reg:
            found['regno'] = m_reg.group(1)

        m_title = re.search(patterns['title'], text)
        if m_title:
            found['title'] = m_title.group(2) if m_title.lastindex and m_title.lastindex >= 2 else m_title.group(1)

        m_hours = re.search(patterns['hours'], text)
        if m_hours:
            found['hours'] = f"{m_hours.group(2)}시간" if m_hours.lastindex and m_hours.lastindex >= 2 else m_hours.group(0)

        return found

    def _run_ocr(self, image_np):
        """
        PaddleOCR 호출 후 EasyOCR 호환 형태로 변환:
        결과 형태: [(bbox, text, conf), ...]
        """
        texts = []
        confs = []
        results = []
        try:
            ocr_result = self.reader.ocr(image_np, cls=True)
            if ocr_result and len(ocr_result) > 0:
                for line in ocr_result[0]:
                    # line: [bbox, (text, score)]
                    if len(line) >= 2:
                        bbox = line[0]
                        text = line[1][0]
                        conf = line[1][1]
                        results.append((bbox, text, conf))
                        texts.append(text)
                        confs.append(conf)
        except Exception as e:
            logger.warning(f"PaddleOCR 호출 실패: {e}")

        return results, texts, confs


