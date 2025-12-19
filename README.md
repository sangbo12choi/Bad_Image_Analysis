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
├── test_chipping_detection.py # Chipping 테스트
├── test_crack_detection.py    # Crack 테스트
├── test_scratch_detection.py  # Scratch 테스트
├── run_scratch_detection.py   # Scratch 간단 실행 스크립트
└── main.py                    # CLI 실행 스크립트
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

### 방법 1: CLI를 통한 실행 (권장)

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