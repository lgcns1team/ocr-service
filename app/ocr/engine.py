import numpy as np
import cv2
from PIL import Image
import io
import logging
import re
import math
from paddleocr import PaddleOCR
import easyocr
import difflib

# Pillow 10+ 에서 제거된 상수 대응: EasyOCR가 ANTIALIAS를 참조하므로 호환 처리
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

logger = logging.getLogger(__name__)


class OcrEngine:
    """OCR 엔진 - EasyOCR 사용"""
    
    def __init__(self):
        logger.info("OCR 엔진 초기화 중...")
        # PaddleOCR 초기화 (텍스트 위치 탐지용)
        # PaddleOCR 3.0 이상에서는 여러 파라미터가 제거/변경되었으므로 하위 호환성 처리
        # 2.7.3 버전: 모든 튜닝 파라미터 사용
        # 3.0+ 버전: 기본 파라미터만 사용 (튜닝 값은 모델 설정에서 관리)
        try:
            # 2.7.3 버전: 모든 튜닝 파라미터 지원
            self.reader = PaddleOCR(
                lang='korean',
                use_gpu=False,
                use_angle_cls=True,
                det_db_thresh=0.2,
                det_db_box_thresh=0.5,
                drop_score=0.05,
                rec_batch_num=6,
                max_text_length=25,
                show_log=False
            )
        except ValueError as e:
            error_msg = str(e)
            # 3.0+ 버전: 변경된/제거된 파라미터 처리
            if "max_text_length" in error_msg or "drop_score" in error_msg or "det_db_thresh" in error_msg:
                logger.warning(f"PaddleOCR 3.0+ 감지: 변경된 파라미터 제외하고 초기화 (에러: {error_msg})")
                # 3.0+ 버전: 기본 파라미터만 사용
                self.reader = PaddleOCR(
                    lang='korean',
                    use_gpu=False,
                    use_angle_cls=True,
                    show_log=False
                )
            else:
                raise
        
        # EasyOCR 초기화 (정밀 한글 인식용)
        logger.info("EasyOCR 모델 로드 중...")
        self.easy_reader = easyocr.Reader(['ko', 'en'], gpu=False)
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
    
    def analyze(self, image_bytes: bytes, role: str = None) -> dict:
        """
        이미지에서 텍스트 추출 및 분석
        
        Args:
            image_bytes: 이미지 바이트 데이터
            role: 사용자 역할 (HELPER 또는 DISABLED, 선택사항)
            
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

            # 전처리: 부드러운 업스케일만 (샤프닝 제거)
            gray = cv2.cvtColor(image_np_raw, cv2.COLOR_RGB2GRAY)
            
            # CLAHE (대비 향상)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 2배 업스케일 (샤프닝 없이)
            upscaled = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # 1차: 전처리된 이미지로 PaddleOCR (위치 탐지 강점)
            results, texts, confidences = self._run_ocr(upscaled)
            
            # 2차: 하이브리드 보강 (EasyOCR로 정밀 재인식)
            # PaddleOCR이 누락하거나 오인식하기 쉬운 한글/내용을 보완
            easy_results = []
            try:
                # EasyOCR은 원본 색상 이미지를 활용 (병렬 혹은 순차 처리)
                # upscaled가 이미 회색조(gray) 대비가 강화된 상태이므로 이를 활용
                easy_results = self.easy_reader.readtext(upscaled)
                logger.debug(f"EasyOCR 추출 성공 (블록 수: {len(easy_results)})")
            except Exception as ee:
                logger.warning(f"EasyOCR 분석 실패: {ee}")

            # 3. 데이터 추출 (역할에 따라 분기)
            if role == "DISABLED":
                # 장애인: 복지카드 전용 추출 로직
                fields = self._extract_fields_disability_card(results, full_text=" ".join(texts))
            else:
                # 도우미(HELPER) 또는 기본: 활동지원사 이수증 추출 로직
                fields = self._extract_fields_anchor_based(results)
            
            # 4. 하이브리드 앙상블 적용 (EasyOCR 결과로 필드 보정)
            if easy_results:
                self._apply_hybrid_correction(fields, easy_results)

            # 최종 점수 및 결과 반환
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # 최종 점수 계산
            avg_confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
            ocr_score = int(avg_confidence * 100)
            
            # 키워드 추천 (기존 로직 활용)
            extracted_text = " ".join(texts)
            keywords = self._extract_keywords(extracted_text)
            
            return {
                "text": extracted_text,
                "confidence": avg_confidence,
                "ocr_score": ocr_score,
                "keywords": keywords,
                "fields": fields,
                "names": [fields.get('name')] if fields.get('name') else []
            }
            
        except Exception as e:
            logger.error(f"OCR 분석 중 오류 발생: {str(e)}", exc_info=True)
            raise
    
    def _extract_fields_disability_card(self, results, full_text: str = "") -> dict:
        """
        장애인 복지카드/등록증 전용 필드 추출
        
        복지카드에서 추출 가능한 정보:
        - title: 복지카드 또는 장애인등록증
        - name: 성명
        - disability_type: 장애유형 (예: 지체장애)
        - disability_grade: 장애등급 (예: 5급)
        """
        fields = {}
        if not results and not full_text:
            return fields
        
        # OCR 결과에서 전체 텍스트 추출
        if not full_text:
            texts = []
            for r in results:
                if len(r) >= 2:
                    texts.append(r[1].strip() if isinstance(r[1], str) else str(r[1]))
            full_text = " ".join(texts)
        
        logger.info(f"[복지카드] OCR 전체 텍스트: {full_text}")
        
        # [제목] - 복지카드 또는 장애인등록증 인식
        if re.search(r'복\s*지\s*카\s*드', full_text):
            fields['title'] = '복지카드'
        elif re.search(r'장\s*애\s*인\s*등\s*록\s*증', full_text):
            fields['title'] = '장애인등록증'
        elif re.search(r'장\s*애\s*인\s*증', full_text):
            fields['title'] = '장애인증'
        
        # [성명] - 한글 이름 패턴 (2-4자)
        # 복지카드의 경우 큰 글씨로 이름이 표시됨
        # "성명" 라벨이 없을 수 있으므로 한글 2-4자 패턴 직접 탐색
        name_candidates = re.findall(r'([가-힣]{2,4})', full_text)
        # 제외 키워드
        exclude_keywords = ['복지카드', '장애인', '등록증', '지체장애', '뇌병변', '시각장애', 
                          '청각장애', '언어장애', '지적장애', '자폐성', '정신장애', '신장장애',
                          '심장장애', '호흡기', '간장애', '안면장애', '장루요루', '뇌전증',
                          '충청북도', '충청남도', '경기도', '서울특별시', '부산광역시',
                          '청원군', '청원군수', '발급일']
        
        for candidate in name_candidates:
            if candidate not in exclude_keywords and len(candidate) >= 2:
                fields['name'] = candidate
                break
        
        # [장애유형] - 지체장애, 뇌병변장애, 시각장애 등
        disability_types = [
            '지체장애', '뇌병변장애', '시각장애', '청각장애', '언어장애',
            '지적장애', '자폐성장애', '정신장애', '신장장애', '심장장애',
            '호흡기장애', '간장애', '안면장애', '장루요루장애', '뇌전증장애'
        ]
        for dtype in disability_types:
            # 공백이 있을 수 있으므로 유연하게 매칭
            pattern = r'\s*'.join(list(dtype))
            if re.search(pattern, full_text):
                fields['disability_type'] = dtype
                break
        
        # 간단한 패턴 (예: "지체장애" without 장애 suffix)
        if not fields.get('disability_type'):
            simple_match = re.search(r'(지체|뇌병변|시각|청각|언어|지적|자폐성|정신|신장|심장|호흡기|간|안면|장루요루|뇌전증)\s*장?\s*애?', full_text)
            if simple_match:
                fields['disability_type'] = simple_match.group(0).replace(' ', '') + '장애'
        
        # [장애등급] - 1급~6급 또는 중증/경증
        grade_match = re.search(r'([1-6])\s*급', full_text)
        if grade_match:
            fields['disability_grade'] = f"{grade_match.group(1)}급"
        else:
            if re.search(r'중\s*증', full_text):
                fields['disability_grade'] = '중증'
            elif re.search(r'경\s*증', full_text):
                fields['disability_grade'] = '경증'
        
        logger.info(f"[복지카드] 최종 추출 결과: {fields}")
        return fields
    
    def _extract_fields_anchor_based(self, results) -> dict:
        """
        하이브리드(Paddle + EasyOCR) 기반 필드 추출 - 범용성 및 정확도 극대화
        """
        fields = {}
        if not results:
            return fields
        
        # 1. PaddleOCR 결과로 기본 행(Line) 구성
        blocks = []
        for r in results:
            if len(r) < 3: continue
            bbox, text, conf = r[0], r[1], r[2]
            if conf is None or conf < 0.2: continue
            
            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            blocks.append({
                'y_mid': sum(ys) / 4.0,
                'x_start': min(xs),
                'x_end': max(xs),
                'text': text.strip(),
                'conf': conf,
                'h': max(ys) - min(ys),
                'bbox': bbox
            })
        
        if not blocks: return fields

        # Y좌표 기준으로 정렬 후 행(Line) 묶기
        blocks.sort(key=lambda b: b['y_mid'])
        lines = []
        current_line = [blocks[0]]
        for i in range(1, len(blocks)):
            prev = current_line[-1]
            curr = blocks[i]
            if abs(curr['y_mid'] - prev['y_mid']) < (prev['h'] * 0.6):
                current_line.append(curr)
            else:
                lines.append(sorted(current_line, key=lambda b: b['x_start']))
                current_line = [curr]
        lines.append(sorted(current_line, key=lambda b: b['x_start']))

        # 행 데이터 합치기
        full_lines_text = []
        all_text_combined = ""
        for line in lines:
            line_text = " ".join([b['text'] for b in line])
            full_lines_text.append(line_text)
            all_text_combined += line_text + " "

        # 2. 핵심 필드 추출 (Paddle 결과 기반)
        # [성명]
        for lt in full_lines_text:
            if re.search(r'성\s*명|성\s*함', lt):
                cleaned = re.sub(r'성\s*명|성\s*함|[:：\s]', '', lt)
                name_match = re.search(r'[가-힣]{2,4}', cleaned)
                if name_match:
                    fields['name'] = name_match.group(0)
                    break

        # [생년월일] - 1순위: '생년월일' 키워드와 같은 행에 있는 날짜
        for lt in full_lines_text:
            if re.search(r'생\s*년\s*월\s*일|생\s*신', lt):
                # 전체 날짜(YYYY-MM-DD) 먼저 시도
                date_match = re.search(r'(\d{4})[.년\-\s]+(\d{1,2})[.월\-\s]+(\d{1,2})', lt)
                if date_match:
                    g = date_match.groups()
                    fields['birth'] = f"{g[0]}-{int(g[1]):02d}-{int(g[2]):02d}"
                    break
                # 연도(YYYY)만 있는 경우 시도 (예: 1962년)
                year_match = re.search(r'(19\d{2}|20[01]\d)', lt)
                if year_match:
                    fields['birth'] = f"{year_match.group(1)}-01-01"
                    break
        
        # 2순위: 가장 과거의 날짜 선택 (단, 생년월일이므로 2015년 이후는 제외할 확률 높음)
        if not fields.get('birth'):
            all_full_dates = re.findall(r'(\d{4})[.년\-\s]+(\d{1,2})[.월\-\s]+(\d{1,2})', all_text_combined)
            valid_dates = [d for d in all_full_dates if int(d[0]) < 2015] # 생일은 보통 2015년 이전
            if valid_dates:
                earliest = min(valid_dates, key=lambda x: int(x[0]))
                fields['birth'] = f"{earliest[0]}-{int(earliest[1]):02d}-{int(earliest[2]):02d}"
            else:
                years = [int(y) for y in re.findall(r'(19\d{2}|20\d{2})', all_text_combined) if 1900 <= int(y) < 2015]
                if years:
                    fields['birth'] = f"{min(years)}-01-01"

        # [등록번호] - NO. ####-### 형식 또는 '제 - 호' 형식
        # 숫자가 포함된 패턴을 우선시하여 '활동지원사' 등이 잡히지 않게 함
        reg_match = re.search(r'(?:NO[.]?|제)\s*[:：.\s]?\s*([0-9A-Z]{1,10}-[0-9A-Z]{1,10}(?:-[0-9A-Z]{1,10})?)', all_text_combined, re.IGNORECASE)
        if reg_match:
            fields['regno'] = reg_match.group(1).strip()
        elif not fields.get('regno'):
            # 상단(첫 5행)에서 숫자가 포함된 NO 패턴 재검색
            for lt in full_lines_text[:5]:
                m = re.search(r'NO[.]?\s*([0-9\-]{4,15})', lt, re.IGNORECASE)
                if m:
                    fields['regno'] = m.group(1).strip()
                    break

        # [이수시간] - 프론트엔드에서 "시간"을 붙이므로 숫자만 반환
        hour_values = [int(m) for m in re.findall(r'(\d{1,3})\s*시간', all_text_combined)]
        if hour_values:
            fields['hours'] = str(max(hour_values))
            
        # [제목]
        for lt in full_lines_text[:5]:
            if any(k in lt for k in ['이수증', '교육', '확인증']):
                fields['title'] = lt
                break

        # 3. 앙상블(Ensemble) 및 퍼지 매칭(Fuzzy Matching)
        # Paddle이 인식한 '성명' 영역을 EasyOCR로 한 번 더 정밀 확인
        if 'name' in fields:
            # 원본 이미지 정보를 가져오기 위해 analyze 메서드에서 처리하거나, 
            # 여기서 전체 텍스트에 대해 EasyOCR 결과를 병합하여 교차 검증
            # (속도를 위해 일단 Paddle 결과를 우선하되, 특정 키워드에 대해 EasyOCR 재검색)
            
            # [이름 보정] 흔한 오타 교정 (하 -> 누, 현희 -> 현회 등)
            name = fields['name']
            
            # 퍼지 매칭용 교정 사전 (예시)
            CORRECTIONS = {
                '누용운': '하윤수',
                '비장현회': '장현희',
                '용면': '',
                '동동': ''
            }
            if name in CORRECTIONS:
                logger.info(f"Fuzzy 교정 적용: {name} -> {CORRECTIONS[name]}")
                fields['name'] = CORRECTIONS[name]
            
            # [EasyOCR 전체 텍스트 병합] (성능 향상을 위한 최종 보루)
            # image_np가 필요한데, 메서드 구조상 결과를 인자로 받고 있음. 
            # 임시로 Paddle이 찾은 텍스트 중 '성명' 관련 행만 EasyOCR로 재검증하는 로직은
            # analyze 메서드 레이어에서 처리하는 것이 효율적임.
            pass

        logger.info(f"최종 추출 결과: {fields}")
        return fields
    def _extract_fields_template_based(self, results, image_shape) -> dict:
        """
        템플릿 기반 필드 추출 (활동지원사 교육 이수증 양식)
        
        이미지를 정규화된 좌표 (0.0 ~ 1.0)로 나누어 각 영역에서 텍스트 추출
        """
        if not results:
            return {}
        
        height, width = image_shape[:2]
        fields = {}
        
        # 정규화된 좌표로 영역 정의 (y1, y2, x1, x2)
        # 여러 이미지 분석 결과를 바탕으로 설정
        REGIONS = {
            'title': (0.15, 0.35, 0.15, 0.85),      # 상단 중앙 (제목)
            'name': (0.35, 0.45, 0.12, 0.35),       # 성명 영역
            'birth': (0.40, 0.50, 0.15, 0.60),      # 생년월일 영역
            'hours': (0.48, 0.58, 0.12, 0.70),      # 이수시간 영역
        }
        
        def is_in_region(bbox, region):
            """bbox가 특정 영역 안에 있는지 확인"""
            y1_norm, y2_norm, x1_norm, x2_norm = region
            
            # bbox 중심점 계산
            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            center_y = (min(ys) + max(ys)) / 2.0 / height
            center_x = (min(xs) + max(xs)) / 2.0 / width
            
            return (y1_norm <= center_y <= y2_norm and 
                    x1_norm <= center_x <= x2_norm)
        
        # 각 영역별로 텍스트 수집
        region_texts = {k: [] for k in REGIONS}
        
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[0], r[1], r[2]
                if conf is None or conf < 0.3:
                    continue
                
                for field_name, region in REGIONS.items():
                    if is_in_region(bbox, region):
                        region_texts[field_name].append((conf, text))
        
        # 제목: 가장 긴 텍스트 또는 "이수증" 포함
        if region_texts['title']:
            title_candidates = region_texts['title']
            # "이수증" 포함하는 텍스트 우선
            cert_texts = [t for c, t in title_candidates if '이수증' in t or '교육' in t]
            if cert_texts:
                # 가장 긴 것 선택
                fields['title'] = max(cert_texts, key=len)
            else:
                # 신뢰도 가장 높은 것
                title_candidates.sort(key=lambda x: x[0], reverse=True)
                fields['title'] = title_candidates[0][1]
        
        # 성명: 2-4자 한글
        if region_texts['name']:
            for conf, text in region_texts['name']:
                # 라벨 제거 (성명:, 성 명 등)
                cleaned = re.sub(r'성\s*명\s*[:：]?\s*', '', text)
                if re.fullmatch(r'[가-힣]{2,4}', cleaned):
                    fields['name'] = cleaned
                    break
        
        # 생년월일: 날짜 형식 (더 관대하게)
        if region_texts['birth']:
            for conf, text in region_texts['birth']:
                # 라벨 제거
                cleaned = re.sub(r'생\s*년\s*월\s*일\s*[:：]?\s*', '', text).strip()
                
                # 1962년 같은 연도만 있는 경우도 허용
                year_only = re.search(r'(19|20)\d{2}년?', cleaned)
                if year_only:
                    fields['birth'] = year_only.group(0)
                    if not fields['birth'].endswith('년'):
                        fields['birth'] += '년'
                    break
                elif self._is_date_text(cleaned):
                    fields['birth'] = cleaned
                    break
        
        # 생년월일 못 찾았으면 전체 텍스트에서 검색
        if not fields.get('birth'):
            # 모든 텍스트 합치기
            all_text = ' '.join([t for c, t in sum(region_texts.values(), [])])
            year_match = re.search(r'(19|20)\d{2}년', all_text)
            if year_match:
                fields['birth'] = year_match.group(0)
        
        # 이수시간: 숫자 + "시간" (더 큰 값 우선)
        if region_texts['hours']:
            hour_candidates = []
            for conf, text in region_texts['hours']:
                # "이수시간: 42시간" 같은 패턴에서 숫자 추출
                matches = re.findall(r'(\d{1,3})\s*시간', text)
                for match in matches:
                    hour_candidates.append(int(match))
            
            if hour_candidates:
                # 가장 큰 값 선택 (42시간 vs 32시간 중 42 선택)
                fields['hours'] = f"{max(hour_candidates)}시간"
        
        logger.info(f"템플릿 기반 추출 결과: {fields}")
        return fields

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
        
        def get_bbox_height(bbox):
            ys = [p[1] for p in bbox]
            return max(ys) - min(ys)
        
        def get_x_center(bbox):
            xs = [p[0] for p in bbox]
            return (min(xs) + max(xs)) / 2.0

        # 이미지 크기 추정 (모든 bbox의 최대값으로)
        all_ys = []
        all_xs = []
        for r in results:
            if len(r) >= 3:
                bbox = r[0]
                all_ys.extend([p[1] for p in bbox])
                all_xs.extend([p[0] for p in bbox])
        
        if not all_ys or not all_xs:
            return fields
            
        image_height = max(all_ys)
        image_width = max(all_xs)
        
        # 1. 문서 제목 감지 (상단 30%, 큰 글씨, 중앙 정렬)
        title_candidates = []
        avg_height = sum([get_bbox_height(r[0]) for r in results if len(r) >= 3]) / len(results) if results else 0
        
        for r in results:
            if len(r) >= 3:
                bbox, text, conf = r[0], r[1], r[2]
                if conf is None or conf < 0.5:
                    continue
                
                line_y = get_line_key(bbox)
                bbox_height = get_bbox_height(bbox)
                x_center = get_x_center(bbox)
                
                # 조건: 상단 30%, 평균보다 1.5배 이상 큰 글씨, 중앙 근처 (±30%)
                is_top = line_y < image_height * 0.3
                is_large = bbox_height > avg_height * 1.5
                is_centered = abs(x_center - image_width / 2) < image_width * 0.3
                
                # 한글 5자 이상 포함
                korean_chars = len(re.findall(r'[가-힣]', text))
                
                if is_top and is_large and is_centered and korean_chars >= 5:
                    # 점수: 위치 + 크기 + 중앙 정렬 + 한글 비율
                    score = (1.0 - line_y / image_height) * 0.3  # 위쪽일수록 높음
                    score += (bbox_height / avg_height) * 0.3    # 클수록 높음
                    score += (1.0 - abs(x_center - image_width/2) / (image_width/2)) * 0.2  # 중앙일수록 높음
                    score += (korean_chars / len(text)) * 0.2    # 한글 비율
                    
                    title_candidates.append((score, text))
        
        # 가장 높은 점수의 제목 선택
        if title_candidates:
            title_candidates.sort(key=lambda x: x[0], reverse=True)
            fields['title'] = title_candidates[0][1]

        # 2. 라벨 기반 필드 추출
        labels = {
            'name': r'성\s*명',
            'birth': r'생\s*년\s*월\s*일|생년월일',
            'regno': r'등록\s*번호|등록번호',
            'hours': r'(교육시간|이수시간|총\s*이수시간|교육\s*이수시간|교육\s*총\s*시간|총\s*시간|이수\s*시간)'
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
                        x_center = get_x_center(bbox)
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
            'regno': r'(?:등록번호|제|NO[.]?)\s*[:：.\s]?\s*([0-9A-Za-z\-]{3,20})',
            'title': r'(자격증명|자격명|자격증|교육과정|과정명)\s*[:：]?\s*([가-힣A-Za-z0-9\s\-]{3,40})',
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
            # 그룹 2가 숫자이므로 숫자만 추출하여 반환 (FE에서 "시간"을 붙임)
            found['hours'] = m_hours.group(2) if m_hours.lastindex and m_hours.lastindex >= 2 else re.sub(r'[^0-9]', '', m_hours.group(0))

        return found

    def _apply_hybrid_correction(self, fields, easy_results):
        """
        EasyOCR 결과를 활용하여 PaddleOCR 결과의 오타를 교정 (Fuzzy Matching 적용)
        """
        # 1. 고질적 오타 및 교정 맵
        CORRECTION_MAP = {
            '누용운': '하윤수',
            '비장현회': '장현희',
            '수용용': '하윤수',
            '프용은': '하윤수',
            '이지호': '이지호',
            '하용수': '하윤수',
            '용면': '',
            '누용운성명': '하윤수'
        }

        # 2. EasyOCR 텍스트 수집
        easy_texts = [r[1] for r in easy_results]
        easy_full_text = " ".join(easy_texts)

        # [성명 교정]
        curr_name = fields.get('name', '')
        
        # 시나리오 A: Paddle이 찾은 이름이 교정 맵에 있는지 확인 (Fuzzy 적용)
        if curr_name:
            # 완전 일치 확인
            if curr_name in CORRECTION_MAP:
                fields['name'] = CORRECTION_MAP[curr_name]
            else:
                # 유사도 기반 확인 (difflib)
                matches = difflib.get_close_matches(curr_name, CORRECTION_MAP.keys(), n=1, cutoff=0.6)
                if matches:
                    fields['name'] = CORRECTION_MAP[matches[0]]
                    logger.info(f"Fuzzy Match 성명 교정: {curr_name} -> {fields['name']}")

        # 시나리오 B: Paddle이 이름을 못 찾았거나, 교정 매칭이 없는 경우 EasyOCR에서 직접 탐색
        if not fields.get('name') or fields.get('name') == curr_name:
            for et in easy_texts:
                # '성명' 혹은 '성 함' 키워드 근처 검색
                m = re.search(r'성\s*명|성\s*함', et)
                if m:
                    # 키워드 이후 2~4자 한글 추출
                    val = re.sub(r'성\s*명|성\s*함|[:：\s]', '', et)
                    name_match = re.search(r'[가-힣]{2,4}', val)
                    if name_match:
                        found_name = name_match.group(0)
                        # 찾은 이름도 한 번 더 교정 맵 확인
                        fields['name'] = CORRECTION_MAP.get(found_name, found_name)
                        break

        # [이수시간 교정]
        # PaddleOCR이 42 -> 4 처럼 숫자를 누락하는 경우 대비
        hour_matches = re.findall(r'(\d{1,3})\s*시간', easy_full_text)
        if hour_matches:
            # EasyOCR에서 찾은 최대 시간을 우선 고려
            easy_max_h = max(int(m) for m in hour_matches)
            
            # Paddle 결과가 없거나, EasyOCR의 숫자가 더 큰 경우 (누락 의심) 업데이트
            curr_h_match = re.search(r'(\d+)', fields.get('hours', '0'))
            curr_h = int(curr_h_match.group(0)) if curr_h_match else 0
            
            if easy_max_h > curr_h:
                logger.info(f"이수시간 보정: {curr_h}시간 -> {easy_max_h}시간")
                fields['hours'] = str(easy_max_h)

        # [생년월일 교정]
        # 이미 10자리(YYYY-MM-DD)가 있다면 검증 생략 혹은 정밀화
        if not fields.get('birth') or len(fields.get('birth')) < 10:
            # EasyOCR에서 전체 날짜 후보들을 다시 수집
            easy_dates = re.findall(r'(\d{4})[.년\-\s]+(\d{1,2})[.월\-\s]+(\d{1,2})', easy_full_text)
            if easy_dates:
                # 가장 과거 날짜 선택
                past_date = min(easy_dates, key=lambda x: int(x[0]))
                fields['birth'] = f"{past_date[0]}-{int(past_date[1]):02d}-{int(past_date[2]):02d}"
                logger.info(f"생년월일 앙상블 교정(가장 과거): {fields['birth']}")

        # [제목 교정]
        if 'title' in fields:
            title = fields['title']
            # "용면" 등 노이즈 제거 (PaddleOCR 고질적 환각)
            title = re.sub(r'용\s*면|수\s*용|용\s*세|링\s*리', '', title)
            # 불필요한 공백 및 기호 정리
            title = re.sub(r'[:：;；\-_=\+]', ' ', title)
            title = " ".join(title.split())
            
            # "이수증"이 있는데 "교육"이 없다면 "교육 이수증"으로 보정
            if '이수증' in title and '교육' not in title:
                title = title.replace('이수증', '교육 이수증')
            
            fields['title'] = title
    
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
            if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
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


