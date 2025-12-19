# Panel Hard Defect Analysis System

디스플레이 패널의 Hard Defect 불량을 이미지 분석을 통해 자동으로 감지하고 분류하는 시스템입니다.

## 프로젝트 구조

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
├── scratch_generator.py        # Scratch 이미지 생성기
├── defect_analyzer.py         # 결함 분석 모듈
├── analyze_gui.py             # GUI 프로그램 ⭐ (가장 쉬움)
├── analyze_console.py         # 인터랙티브 콘솔 프로그램
├── main.py                    # 기본 CLI 실행 스크립트
├── test_chipping_detection.py # Chipping 테스트
├── test_crack_detection.py    # Crack 테스트
├── test_scratch_detection.py  # Scratch 테스트
└── run_scratch_detection.py   # Scratch 간단 실행 스크립트
```

## 주요 기능

- **이미지 전처리**: 노이즈 제거, 대비 향상, 이진화
- **결함 감지**: 다양한 알고리즘을 통한 결함 영역 탐지
- **결함 분류**: 크기, 형태, 위치 기반 분류
- **Chipping 감지**: 가장자리/모서리 Chipping 특화 감지 알고리즘
- **Crack 감지**: 외곽에서 시작하는 선형 균열 특화 감지 알고리즘
- **Scratch 감지**: 패널 표면 어디서나 발생하는 선형 긁힘 특화 감지 알고리즘
- **가상 이미지 생성**: 테스트용 Chipping/Crack/Scratch 불량 이미지 생성기
- **결과 시각화**: 감지된 결함을 시각적으로 표시
- **리포트 생성**: 분석 결과를 CSV/이미지로 저장

## 설치 방법

### 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

필요한 패키지:
- opencv-python (이미지 처리)
- numpy (수치 계산)
- scikit-image (이미지 분석)
- matplotlib (시각화)
- pandas (데이터 처리)
- scipy (과학 계산)
- Pillow (이미지 I/O)

## 실행 방법

### 방법 1: GUI 프로그램 (가장 쉬움) ⭐⭐⭐

마우스 클릭만으로 쉽게 사용할 수 있는 그래픽 인터페이스입니다.

```bash
python analyze_gui.py
```

#### 주요 기능
- 🖱️ 마우스 클릭만으로 모든 기능 사용
- 📋 직관적인 메뉴 인터페이스
- 🔍 불량 유형별 선택 분석
- 📊 실시간 결과 시각화 (이미지, 통계, 리포트)
- 💾 원클릭 결과 저장

자세한 사용법은 [GUI_USAGE.md](GUI_USAGE.md)를 참조하세요.

### 방법 2: 인터랙티브 콘솔 프로그램 ⭐⭐

불량 유형별로 선택하여 분석할 수 있는 인터랙티브 프로그램입니다.

```bash
python analyze_console.py
```

#### 주요 기능
- 📋 메뉴 기반 인터페이스
- 🔍 불량 유형별 선택 분석 (Chipping, Crack, Scratch, Bubble)
- 📁 단일 이미지 및 배치 처리 지원
- ⚙️ 분석 설정 변경 가능

자세한 사용법은 [CONSOLE_USAGE.md](CONSOLE_USAGE.md)를 참조하세요.

#### 빠른 시작 예시

```bash
# 인터랙티브 모드 실행
python analyze_console.py

# 명령줄에서 직접 실행 (Chipping만 분석)
python analyze_console.py image.jpg -o results --type chipping

# 배치 처리 (Crack만 분석)
python analyze_console.py input_folder/ -o output/ --type crack
```

### 방법 2: 기본 CLI (main.py)

#### 단일 이미지 분석

```bash
python main.py 이미지경로.jpg -o 출력폴더
```

예시:
```bash
python main.py panel_image.jpg -o results
```

#### 배치 처리 (폴더 내 모든 이미지)

```bash
python main.py 입력폴더/ -o 출력폴더/
```

예시:
```bash
python main.py input_images/ -o output_results/
```

#### 옵션 설명

- `-o, --output`: 결과 저장 폴더 (기본값: output)
- `--min-area`: 최소 결함 크기 (기본값: 10)
- `--max-area`: 최대 결함 크기 (기본값: 100000)
- `--threshold`: 이진화 방법 (adaptive/otsu/manual, 기본값: adaptive)
- `--no-show`: 결과 이미지를 화면에 표시하지 않음

예시:
```bash
python main.py image.jpg -o results --min-area 50 --threshold otsu
```

### 방법 2: Python 코드로 실행

#### 기본 사용법

```python
from defect_analyzer import DefectAnalyzer

# 분석기 생성
analyzer = DefectAnalyzer()

# 이미지 분석
defects = analyzer.analyze_image('panel_image.jpg')

# 결과 시각화
analyzer.visualize_results('panel_image.jpg', defects, save_path='result.jpg')

# 리포트 생성
analyzer.generate_report(defects, 'report.csv')
```

#### Chipping 감지 예제

```python
from chipping_generator import ChippingImageGenerator
from defect_analyzer import DefectAnalyzer

# 1. 가상 Chipping 이미지 생성
generator = ChippingImageGenerator(width=1920, height=1080)
panel = generator.generate_panel_with_chipping(
    num_chippings=2,
    chipping_types=['corner', 'edge']
)
generator.save_image(panel, 'chipping_panel.jpg')

# 2. Chipping 감지
analyzer = DefectAnalyzer()
defects = analyzer.analyze_image('chipping_panel.jpg')

# Chipping만 필터링
chippings = [d for d in defects if d.defect_type == 'chipping']
print(f"감지된 Chipping: {len(chippings)}개")

# 결과 시각화
analyzer.visualize_results('chipping_panel.jpg', defects, save_path='result.jpg')
```

### 방법 3: 테스트 스크립트 실행

#### Chipping 감지 테스트

```bash
python test_chipping_detection.py
```

이 스크립트는:
1. 다양한 유형의 가상 Chipping 이미지를 생성
2. 각 이미지에서 Chipping을 감지
3. 결과를 시각화하여 저장

#### 예제 코드 실행

```bash
python example_usage.py
```

## 사용 방법

### 기본 사용법

```python
from defect_analyzer import DefectAnalyzer

analyzer = DefectAnalyzer()
results = analyzer.analyze_image('path/to/panel_image.jpg')
analyzer.visualize_results(results, save_path='result.jpg')
```

### 배치 처리

```python
from defect_analyzer import DefectAnalyzer

analyzer = DefectAnalyzer()
analyzer.batch_analyze('input_folder/', 'output_folder/')
```

## 결함 유형

- **점 결함 (Point Defect)**: 작은 크기의 점 형태 결함
- **선 결함 (Line Defect)**: 선 형태의 결함
- **면 결함 (Area Defect)**: 넓은 영역의 결함
- **엣지 결함 (Edge Defect)**: 패널 가장자리 결함
- **Chipping 결함**: 패널 가장자리/모서리의 깨짐/파손 불량
- **Crack 결함**: 패널 외곽에서 시작하는 선형 균열 불량
- **Scratch 결함**: 패널 표면 어디서나 발생하는 선형 긁힘 불량

## Chipping 감지 기능

이 프로젝트는 Display Panel의 Chipping 불량을 특별히 감지할 수 있는 기능을 제공합니다.

### 가상 Chipping 이미지 생성

```python
from chipping_generator import ChippingImageGenerator

generator = ChippingImageGenerator(width=1920, height=1080)

# Chipping이 있는 패널 생성
panel = generator.generate_panel_with_chipping(
    num_chippings=2,
    chipping_types=['corner', 'edge']
)
generator.save_image(panel, 'chipping_panel.jpg')
```

### Chipping 감지

```python
from defect_analyzer import DefectAnalyzer

analyzer = DefectAnalyzer()
defects = analyzer.analyze_image('chipping_panel.jpg')

# Chipping만 필터링
chippings = [d for d in defects if d.defect_type == 'chipping']
print(f"감지된 Chipping: {len(chippings)}개")
```

### 테스트 실행

```bash
python test_chipping_detection.py
```

## Crack 감지 기능

이 프로젝트는 Display Panel의 Crack(균열) 불량을 특별히 감지할 수 있는 기능을 제공합니다.
Crack은 Panel의 외곽(가장자리)에서만 발생합니다.

### 가상 Crack 이미지 생성

```python
from crack_generator import CrackImageGenerator

generator = CrackImageGenerator(width=1920, height=1080)

# Crack이 있는 패널 생성 (외곽에서 시작)
panel = generator.generate_panel_with_crack(
    num_cracks=2,
    crack_types=['straight', 'curved']  # 'straight', 'curved', 'branching'
)
generator.save_image(panel, 'crack_panel.jpg')
```

### Crack 감지

```python
from defect_analyzer import DefectAnalyzer

analyzer = DefectAnalyzer()
defects = analyzer.analyze_image('crack_panel.jpg')

# Crack만 필터링
cracks = [d for d in defects if d.defect_type == 'crack']
print(f"감지된 Crack: {len(cracks)}개")

# 결과 시각화 (Crack은 청록색으로 표시됨)
analyzer.visualize_results('crack_panel.jpg', defects, save_path='result.jpg')
```

### Crack 감지 테스트

```bash
python test_crack_detection.py
```

이 스크립트는:
1. 다양한 유형의 가상 Crack 이미지를 생성 (외곽에서 시작)
2. 각 이미지에서 Crack을 감지
3. 결과를 시각화하여 저장

### Crack 감지 특징

- **위치 기반**: 외곽(이미지 경계의 5% 이내)에서 시작하는 결함만 감지
- **형태 분석**: 선형적인 형태 (종횡비 > 8 또는 < 0.125) 감지
- **선형 구조**: 높은 perimeter/area 비율로 선형 구조 식별
- **크기 필터링**: 50~10,000 픽셀 범위의 결함만 Crack으로 분류

## 폴더 구조

프로젝트는 다음과 같은 폴더 구조를 사용합니다:

- **`generated_images/`**: 가상으로 생성된 불량 이미지
  - `chipping/`: Chipping 샘플 이미지
  - `crack/`: Crack 샘플 이미지
  - `scratch/`: Scratch 샘플 이미지
- **`test_results/`**: 테스트 실행 결과
  - `chipping/`: Chipping 테스트 결과 이미지
  - `crack/`: Crack 테스트 결과 이미지
  - `scratch/`: Scratch 테스트 결과 이미지
- **`output/`**: 일반 분석 결과 (CLI 사용 시)

### 기존 이미지 정리

루트 디렉토리에 있는 기존 이미지 파일들을 새 폴더 구조로 정리하려면:

```bash
python organize_images.py
```

## 파라미터 조정

`defect_analyzer.py`의 설정값을 조정하여 감지 민감도를 변경할 수 있습니다.

## RED단계 시작!!!

## GREEN 단계 시작!!!

## 리팩토링 목록

코드 품질 개선을 위한 리팩토링 작업 목록입니다. 상세 분석은 [CODE_ANALYSIS_REPORT.md](CODE_ANALYSIS_REPORT.md)를 참조하세요.

### 🔴 높은 우선순위 (High Priority)

#### 1. 중복 코드 제거
- [x] KOREAN_FONT_PROP 체크 헬퍼 메서드 생성
  - `analyze_gui.py`에서 10회 이상 반복되는 폰트 체크 로직 통합
  - `_add_text_with_font()`, `_set_title_with_font()`, `_set_label_with_font()` 헬퍼 메서드 생성 완료
- [x] 색상 맵 상수화
  - `display_image()`, `create_highlighted_image()`, `visualize_results()`에서 중복 정의된 색상 맵을 상수로 통합
  - `DEFECT_COLOR_MAP`, `DEFECT_COLOR_MAP_RGB` 상수 정의 완료
- [x] 이미지 초기화 텍스트 상수화
  - "원본 이미지\n(이미지를 선택하세요)", "분석 결과\n(분석을 실행하세요)" 텍스트를 상수로 정의
  - `TEXT_ORIGINAL_IMAGE_PLACEHOLDER`, `TEXT_RESULT_IMAGE_PLACEHOLDER` 상수 정의 완료

#### 2. 사용하지 않는 코드 제거
- [x] `detect_bubble()` 메서드 제거 (`defect_analyzer.py`)
  - 사용되지 않는 메서드 제거 완료 (33라인 제거)
- [x] `_is_bubble_defect()` 메서드 제거 (`defect_analyzer.py`)
  - 사용되지 않는 메서드 제거 완료 (30라인 제거)
- [x] 사용하지 않는 import 제거
  - `sys` (analyze_gui.py) - 이미 제거됨
  - `Image, ImageTk` from PIL (analyze_gui.py) - 이미 제거됨

#### 3. 긴 메서드 분리
- [ ] `display_image()` 메서드 분리 (110라인 → 3개 메서드로 분리)
  - 이미지 렌더링 로직 분리
  - 불량 그리기 로직 분리
  - 라벨 표시 로직 분리
- [ ] `classify_defect()` 메서드 분리 (87라인 → 3개 메서드로 분리)
  - 속성 계산 로직 분리
  - 유형 분류 로직 분리
  - 심각도 분류 로직 분리
- [ ] `reset_application()` 메서드 분리 (86라인 → 2개 메서드로 분리)
  - 상태 초기화 로직 분리
  - UI 초기화 로직 분리

### 🟡 중간 우선순위 (Medium Priority)

#### 4. 단일 책임 원칙 (SRP) 개선
- [ ] DefectAnalysisGUI 클래스 분리
  - UI 컴포넌트별로 분리 (ImageDisplay, StatisticsDisplay, ReportDisplay)
  - 비즈니스 로직 분리 (AnalysisController)
- [ ] DefectAnalyzer 클래스 분리
  - 전처리기 분리 (ImagePreprocessor)
  - 감지기 분리 (DefectDetector)
  - 분류기 분리 (DefectClassifier)
  - 시각화 분리 (ResultVisualizer)

#### 5. 개방-폐쇄 원칙 (OCP) 개선
- [ ] 전략 패턴 도입
  - DefectDetectionStrategy 인터페이스 생성
  - ThresholdStrategy 인터페이스 생성
- [ ] 팩토리 패턴 도입
  - DefectClassifierFactory 생성
  - 불량 유형 추가 시 기존 코드 수정 없이 확장 가능하도록

#### 6. 매직 넘버 상수화
- [ ] 임계값 상수 정의
  - `EDGE_THRESHOLD = 0.1`
  - `CORNER_THRESHOLD = 0.15`
  - `CHIPPING_THRESHOLD_MULTIPLIER = 2.0`
- [ ] 크기 범위 상수 정의
  - `CHIPPING_MIN_AREA = 100`, `CHIPPING_MAX_AREA = 50000`
  - `CRACK_MIN_AREA = 50`, `CRACK_MAX_AREA = 10000`
  - `SCRATCH_MIN_AREA = 30`, `SCRATCH_MAX_AREA = 5000`

#### 7. 긴 매개변수 목록 개선
- [ ] DefectProperties 데이터 클래스 생성
  - `_is_chipping_defect()`, `_is_crack_defect()`, `_is_scratch_defect()` 메서드의 매개변수 통합
- [ ] 기본 타입 집착 개선
  - 불량 유형을 Enum으로 변경 (`DefectType` enum)
  - 심각도를 Enum으로 변경 (`Severity` enum)

### 🟢 낮은 우선순위 (Low Priority)

#### 8. 의존성 역전 원칙 (DIP) 개선
- [ ] 인터페이스 도입
  - `IDefectAnalyzer` 인터페이스 생성
  - 추상화를 통한 의존성 역전
- [ ] 의존성 주입 도입
  - `DefectAnalysisGUI`에서 `DefectAnalyzer` 직접 생성 대신 주입받도록 변경

#### 9. 로깅 시스템 도입
- [ ] print 문을 logging으로 교체
  - `logging` 모듈 사용
  - 로그 레벨 설정 (DEBUG, INFO, WARNING, ERROR)
  - 프로덕션 환경에 적합한 로깅

#### 10. 테스트 가능성 개선
- [ ] 비즈니스 로직과 GUI 분리
  - MVC 또는 MVP 패턴 적용
- [ ] 전역 상태 제거
  - `KOREAN_FONT_PROP` 전역 변수를 의존성 주입으로 변경
  - 테스트 격리 가능하도록 개선

#### 11. 성능 최적화
- [ ] 이미지 캐싱
  - 이미지 재로드 방지를 위한 캐싱 메커니즘 도입
- [ ] 메모리 관리
  - 큰 이미지 처리 시 메모리 이슈 해결

### 진행 상황 추적

리팩토링 작업을 진행할 때 위 체크박스를 업데이트하여 진행 상황을 추적하세요.

**참고**: 상세한 분석 내용은 [CODE_ANALYSIS_REPORT.md](CODE_ANALYSIS_REPORT.md) 파일을 참조하세요.