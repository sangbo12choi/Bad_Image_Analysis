# Panel Hard Defect Analysis System - PRD 리포트

## 프로젝트 개요

**프로젝트명**: Panel Hard Defect Analysis System  
**목적**: 디스플레이 패널의 Hard Defect 불량을 이미지 분석을 통해 자동으로 감지하고 분류하는 시스템 개발  
**개발 기간**: 2024년  
**브랜치**: PRD (Product Requirements Document)

---

## 1. 프로젝트 목표

Display Panel 제조 공장에서 발생하는 Hard Defect 불량을 이미지 분석을 통해 자동으로 감지하고 분류하여, 품질 검사 프로세스를 자동화하고 효율성을 향상시키는 것을 목표로 합니다.

### 주요 목표
- 다양한 Hard Defect 유형의 자동 감지
- 가상 불량 이미지 생성 기능 제공
- 정확한 결함 분류 및 심각도 평가
- 결과 시각화 및 리포트 생성

---

## 2. 구현된 기능

### 2.1 결함 감지 기능

#### 기본 결함 유형
- **점 결함 (Point Defect)**: 작은 크기의 점 형태 결함
- **선 결함 (Line Defect)**: 선 형태의 결함
- **면 결함 (Area Defect)**: 넓은 영역의 결함
- **엣지 결함 (Edge Defect)**: 패널 가장자리 결함

#### 특화 결함 감지
- **Chipping 감지**: 패널 가장자리/모서리의 깨짐/파손 불량
  - 위치: 가장자리/모서리 (이미지 크기의 10% 이내)
  - 형태: 불규칙한 형태 (solidity < 0.85)
  - 크기: 100~50,000 픽셀
  
- **Crack 감지**: 패널 외곽에서 시작하는 선형 균열 불량
  - 위치: 외곽(이미지 경계의 5% 이내)에서 시작
  - 형태: 선형 구조 (종횡비 > 8 또는 < 0.125)
  - 크기: 50~10,000 픽셀
  
- **Scratch 감지**: 패널 표면 어디서나 발생하는 선형 긁힘 불량
  - 위치: 패널 표면 어디서나 발생 가능 (외곽 제한 없음)
  - 형태: 선형 구조 (종횡비 > 6 또는 < 0.167)
  - 크기: 30~5,000 픽셀

### 2.2 가상 이미지 생성 기능

#### Chipping 이미지 생성기 (`chipping_generator.py`)
- **모서리 Chipping** (corner): 모서리에서 발생하는 불규칙한 형태
- **가장자리 Chipping** (edge): 가장자리를 따라 발생하는 형태
- **불규칙 Chipping** (irregular): 완전히 불규칙한 형태

#### Crack 이미지 생성기 (`crack_generator.py`)
- **직선 Crack** (straight): 직선 형태의 균열
- **곡선 Crack** (curved): 곡선 형태의 균열
- **분기 Crack** (branching): 분기되는 균열
- **특징**: 외곽(가장자리)에서만 시작

#### Scratch 이미지 생성기 (`scratch_generator.py`)
- **직선 Scratch** (straight): 직선 형태의 긁힘
- **곡선 Scratch** (curved): 곡선 형태의 긁힘
- **지그재그 Scratch** (zigzag): 지그재그 형태의 긁힘
- **특징**: 패널 표면 어디서나 발생 가능

### 2.3 이미지 처리 기능

- **전처리**
  - 노이즈 제거 (가우시안 블러)
  - 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
  - 그레이스케일 변환

- **결함 감지**
  - 적응형/OTSU/수동 이진화
  - 형태학적 연산 (Opening, Closing)
  - 윤곽선 검출

- **결함 분류**
  - 크기 기반 분류
  - 형태 분석 (종횡비, Solidity)
  - 위치 기반 분류
  - 심각도 평가 (minor, moderate, severe)

### 2.4 결과 처리 기능

- **시각화**
  - 결함 유형별 색상 구분
  - 바운딩 박스 표시
  - 중심점 표시
  - 상세 정보 라벨

- **리포트 생성**
  - CSV 형식 리포트
  - 유형별/심각도별 통계
  - 상세 결함 정보

---

## 3. 기술 스택

### 프로그래밍 언어
- **Python 3.x**

### 주요 라이브러리
- **OpenCV** (cv2): 이미지 처리 및 컴퓨터 비전
- **NumPy**: 수치 계산 및 배열 처리
- **scikit-image**: 이미지 분석 알고리즘
- **Matplotlib**: 결과 시각화
- **Pandas**: 데이터 처리 및 리포트 생성
- **SciPy**: 과학 계산
- **Pillow**: 이미지 I/O

### 개발 도구
- **Git**: 버전 관리
- **Markdown**: 문서 작성

---

## 4. 프로젝트 구조

```
Bad_Image_Analysis/
├── generated_images/          # 가상 생성 이미지
│   ├── chipping/             # Chipping 샘플 이미지
│   ├── crack/                 # Crack 샘플 이미지
│   └── scratch/              # Scratch 샘플 이미지
├── test_results/              # 테스트 결과
│   ├── chipping/             # Chipping 테스트 결과
│   ├── crack/                 # Crack 테스트 결과
│   └── scratch/              # Scratch 테스트 결과
├── output/                    # 일반 분석 결과
├── chipping_generator.py      # Chipping 이미지 생성기
├── crack_generator.py         # Crack 이미지 생성기
├── scratch_generator.py       # Scratch 이미지 생성기
├── defect_analyzer.py         # 결함 분석 모듈 (핵심)
├── test_chipping_detection.py # Chipping 테스트
├── test_crack_detection.py    # Crack 테스트
├── test_scratch_detection.py  # Scratch 테스트
├── run_crack_detection.py     # Crack 간단 실행 스크립트
├── run_scratch_detection.py   # Scratch 간단 실행 스크립트
├── main.py                    # CLI 실행 스크립트
├── example_usage.py           # 사용 예제
├── setup_folders.py           # 폴더 구조 설정
├── organize_images.py         # 이미지 정리 스크립트
├── README.md                  # 프로젝트 문서
├── QUICKSTART.md              # 빠른 시작 가이드
├── CRACK_DETECTION_GUIDE.md   # Crack 감지 가이드
└── requirements.txt           # 패키지 의존성
```

---

## 5. 주요 모듈 설명

### 5.1 `defect_analyzer.py` - 핵심 분석 모듈

**클래스**: `DefectAnalyzer`

**주요 메서드**:
- `preprocess_image()`: 이미지 전처리
- `detect_defects()`: 일반 결함 감지
- `detect_chipping()`: Chipping 특화 감지
- `detect_crack()`: Crack 특화 감지
- `detect_scratch()`: Scratch 특화 감지
- `classify_defect()`: 결함 분류
- `analyze_image()`: 전체 분석 파이프라인
- `visualize_results()`: 결과 시각화
- `generate_report()`: 리포트 생성
- `batch_analyze()`: 배치 처리

**데이터 클래스**: `Defect`
- 결함 ID, 바운딩 박스, 면적, 중심점
- 결함 유형, 심각도
- 둘레, 종횡비, Solidity

### 5.2 이미지 생성기 모듈

#### `chipping_generator.py`
- `ChippingImageGenerator` 클래스
- `generate_base_panel()`: 기본 패널 생성
- `create_chipping()`: Chipping 생성
- `generate_panel_with_chipping()`: 완전한 패널 생성

#### `crack_generator.py`
- `CrackImageGenerator` 클래스
- 외곽에서 시작하는 Crack 생성
- 직선/곡선/분기 형태 지원

#### `scratch_generator.py`
- `ScratchImageGenerator` 클래스
- 패널 표면 어디서나 Scratch 생성
- 직선/곡선/지그재그 형태 지원

### 5.3 테스트 모듈

- `test_chipping_detection.py`: Chipping 감지 테스트
- `test_crack_detection.py`: Crack 감지 테스트
- `test_scratch_detection.py`: Scratch 감지 테스트

각 테스트 모듈은:
1. 다양한 유형의 가상 이미지 생성
2. 감지 알고리즘 테스트
3. 결과 시각화 및 저장
4. 리포트 생성

---

## 6. 알고리즘 상세

### 6.1 이미지 전처리

1. **그레이스케일 변환**: BGR → Grayscale
2. **가우시안 블러**: 노이즈 제거 (커널 크기: 5x5)
3. **CLAHE**: 대비 향상 (clipLimit: 2.0, tileGridSize: 8x8)

### 6.2 결함 감지 알고리즘

#### 일반 결함 감지
- **적응형 이진화**: Adaptive Threshold (Gaussian)
- **형태학적 연산**: Opening → Closing
- **윤곽선 검출**: External contours

#### Chipping 감지
- **임계값**: 평균 - 2×표준편차
- **위치 필터**: 가장자리/모서리 영역 (10% 이내)
- **형태 필터**: 불규칙한 형태 (solidity < 0.85)

#### Crack 감지
- **임계값**: 평균 - 1.5×표준편차
- **위치 필터**: 외곽 영역 (5% 이내)
- **선형 구조 강화**: 다양한 방향의 형태학적 연산
- **형태 필터**: 선형 구조 (종횡비 > 8 또는 < 0.125)

#### Scratch 감지
- **임계값**: 평균 - 1.5×표준편차
- **위치 제한 없음**: 패널 표면 어디서나
- **선형 구조 강화**: 다양한 방향의 형태학적 연산
- **형태 필터**: 선형 구조 (종횡비 > 6 또는 < 0.167)

### 6.3 결함 분류 알고리즘

**우선순위**:
1. Scratch (외곽에서 시작하지 않는 선형 구조)
2. Crack (외곽에서 시작하는 선형 구조)
3. Chipping (가장자리/모서리의 불규칙한 형태)
4. Edge (가장자리 결함)
5. Line/Area/Point (기본 분류)

**심각도 평가**:
- **severe**: 전체 면적의 1% 이상
- **moderate**: 전체 면적의 0.1% 이상
- **minor**: 그 외

---

## 7. 사용 방법

### 7.1 설치

```bash
pip install -r requirements.txt
python setup_folders.py
```

### 7.2 기본 사용법

#### CLI 사용
```bash
# 단일 이미지 분석
python main.py image.jpg -o output/

# 배치 처리
python main.py input_folder/ -o output_folder/
```

#### Python 코드 사용
```python
from defect_analyzer import DefectAnalyzer

analyzer = DefectAnalyzer()
defects = analyzer.analyze_image('image.jpg')
analyzer.visualize_results('image.jpg', defects, save_path='result.jpg')
analyzer.generate_report(defects, 'report.csv')
```

### 7.3 특화 기능 사용

#### Chipping 감지
```bash
python test_chipping_detection.py
```

#### Crack 감지
```bash
python run_crack_detection.py
```

#### Scratch 감지
```bash
python run_scratch_detection.py
```

---

## 8. 테스트 결과

### 8.1 생성된 테스트 이미지

- **Chipping**: 5개 테스트 케이스 (corner, edge, irregular, multiple)
- **Crack**: 4개 테스트 케이스 (straight, curved, branching, multiple)
- **Scratch**: 4개 테스트 케이스 (straight, curved, zigzag, multiple)

### 8.2 감지 성능

각 결함 유형별로:
- ✅ 가상 생성 이미지에서 결함 정확히 감지
- ✅ 결함 유형 정확히 분류
- ✅ 위치 및 크기 정보 정확히 추출
- ✅ 시각화 결과 명확하게 표시

### 8.3 결과 파일

- **이미지**: `test_results/{type}/*_result.jpg`
- **리포트**: `test_results/{type}/*_report.csv`

---

## 9. 결함 유형별 색상 구분

| 결함 유형 | 색상 | 설명 |
|---------|------|------|
| Scratch | 주황색 (orange) | 패널 표면 긁힘 |
| Crack | 청록색 (cyan) | 외곽 균열 |
| Chipping | 자홍색 (magenta) | 가장자리 깨짐 |
| Point | 빨간색 (red) | 점 결함 |
| Line | 파란색 (blue) | 선 결함 |
| Area | 노란색 (yellow) | 면 결함 |
| Edge | 초록색 (green) | 가장자리 결함 |

---

## 10. 주요 특징

### 10.1 장점

1. **다양한 결함 유형 지원**: 7가지 결함 유형 감지
2. **특화 알고리즘**: Chipping, Crack, Scratch 각각에 특화된 감지 알고리즘
3. **가상 이미지 생성**: 테스트 및 검증을 위한 가상 불량 이미지 생성
4. **자동화**: 배치 처리 지원으로 대량 이미지 처리 가능
5. **시각화**: 직관적인 결과 시각화
6. **리포트**: 상세한 CSV 리포트 생성

### 10.2 기술적 특징

- **모듈화된 구조**: 각 기능이 독립적인 모듈로 구성
- **확장 가능성**: 새로운 결함 유형 추가 용이
- **파라미터 조정**: 감지 민감도 조정 가능
- **폴더 구조 정리**: 체계적인 파일 관리

---

## 11. 향후 개선 사항

### 11.1 단기 개선

1. **딥러닝 모델 통합**: YOLO, U-Net 등 딥러닝 모델 추가
2. **성능 최적화**: 대용량 이미지 처리 속도 개선
3. **GUI 개발**: 사용자 친화적인 그래픽 인터페이스
4. **실시간 처리**: 웹캠 또는 실시간 스트림 처리

### 11.2 중장기 개선

1. **다양한 결함 유형 추가**: 
   - Bubble (기포)
   - Staining (얼룩)
   - Discoloration (변색)
   
2. **3D 분석**: 깊이 정보를 활용한 3D 결함 분석
3. **자동 학습**: 결함 패턴 자동 학습 기능
4. **클라우드 통합**: 클라우드 기반 대규모 처리

### 11.3 품질 개선

1. **정확도 향상**: False Positive/Negative 감소
2. **속도 개선**: 병렬 처리 및 GPU 활용
3. **사용성 개선**: 더 직관적인 인터페이스
4. **문서화 강화**: API 문서 및 사용자 매뉴얼

---

## 12. 프로젝트 통계

### 12.1 코드 통계

- **총 파일 수**: 약 20개
- **주요 모듈**: 4개 (defect_analyzer, chipping_generator, crack_generator, scratch_generator)
- **테스트 스크립트**: 3개
- **실행 스크립트**: 2개
- **문서**: 4개 (README, QUICKSTART, CRACK_DETECTION_GUIDE, Report-PRD)

### 12.2 기능 통계

- **지원 결함 유형**: 7가지
- **특화 감지 알고리즘**: 3개 (Chipping, Crack, Scratch)
- **이미지 생성 유형**: 9가지 (각 결함 유형별 3가지)
- **이진화 방법**: 3가지 (adaptive, otsu, manual)

---

## 13. 사용 사례

### 13.1 제조 공정 검사

- 생산 라인에서 자동 품질 검사
- 불량 패널 조기 발견
- 품질 통계 수집

### 13.2 연구 및 개발

- 새로운 결함 유형 연구
- 알고리즘 성능 평가
- 데이터셋 생성

### 13.3 교육 및 훈련

- 품질 검사원 교육
- 결함 인식 훈련
- 시각적 가이드 제공

---

## 14. 결론

본 프로젝트는 Display Panel의 Hard Defect를 자동으로 감지하고 분류하는 종합적인 시스템을 구현했습니다. 특히 Chipping, Crack, Scratch 세 가지 주요 결함 유형에 대한 특화된 감지 알고리즘을 개발하여 높은 정확도를 달성했습니다.

가상 이미지 생성 기능을 통해 테스트 및 검증이 용이하며, 배치 처리 기능으로 대량의 이미지를 효율적으로 처리할 수 있습니다. 향후 딥러닝 모델 통합 및 성능 최적화를 통해 더욱 발전시킬 수 있을 것으로 기대됩니다.

---

## 15. 참고 자료

### 15.1 문서
- `README.md`: 프로젝트 전체 문서
- `QUICKSTART.md`: 빠른 시작 가이드
- `CRACK_DETECTION_GUIDE.md`: Crack 감지 상세 가이드

### 15.2 코드 예제
- `example_usage.py`: 기본 사용 예제
- `run_crack_detection.py`: Crack 감지 실행 예제
- `run_scratch_detection.py`: Scratch 감지 실행 예제

### 15.3 테스트
- `test_chipping_detection.py`: Chipping 테스트
- `test_crack_detection.py`: Crack 테스트
- `test_scratch_detection.py`: Scratch 테스트

---

**작성일**: 2024년  
**버전**: 1.0  
**브랜치**: PRD  
**작성자**: Development Team

