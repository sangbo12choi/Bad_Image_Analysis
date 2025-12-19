# Crack 불량 이미지 생성 및 감지 가이드

Display Panel의 Crack(균열) 불량을 가상으로 생성하고 감지하는 방법을 안내합니다.

## 빠른 시작

### 방법 1: 테스트 스크립트 실행 (가장 간단)

```bash
python test_crack_detection.py
```

이 명령어 하나로 다음이 자동으로 실행됩니다:
1. 다양한 유형의 Crack 이미지 생성
2. 각 이미지에서 Crack 감지
3. 결과 시각화 및 저장

**생성되는 파일:**
- `generated_images/crack/test_*.jpg` - 생성된 Crack 이미지
- `test_results/crack/test_*_result.jpg` - 감지 결과 이미지

---

## 상세 사용 방법

### 1단계: Crack 이미지 생성

#### Python 코드로 생성

```python
from crack_generator import CrackImageGenerator
from pathlib import Path

# 출력 폴더 생성
output_dir = Path('generated_images/crack')
output_dir.mkdir(parents=True, exist_ok=True)

# 생성기 초기화
generator = CrackImageGenerator(width=1920, height=1080)

# Crack 이미지 생성
panel = generator.generate_panel_with_crack(
    num_cracks=2,  # 생성할 Crack 개수
    crack_types=['straight', 'curved'],  # 유형: 'straight', 'curved', 'branching'
    min_length=150,  # 최소 길이
    max_length=400   # 최대 길이
)

# 이미지 저장
output_path = output_dir / 'my_crack_panel.jpg'
generator.save_image(panel, str(output_path))
print(f"이미지 저장 완료: {output_path}")
```

#### Crack 유형 설명

- **`straight`**: 직선 형태의 균열
- **`curved`**: 곡선 형태의 균열
- **`branching`**: 분기되는 균열

#### 직접 Crack 생성 (세밀한 제어)

```python
import math
from crack_generator import CrackImageGenerator

generator = CrackImageGenerator(width=1920, height=1080)
panel = generator.generate_base_panel()

# 상단 가장자리에서 시작하는 직선 Crack
panel = generator.create_crack(
    panel,
    start_point=(200, 10),      # 시작 위치 (x, y) - 외곽에 위치
    length=300,                  # Crack 길이
    direction=math.pi / 2,       # 방향 (라디안, π/2 = 아래쪽)
    crack_type='straight',       # 유형
    width=2,                     # 두께 (픽셀)
    depth=0.5                    # 깊이/어둠 정도 (0~1)
)

generator.save_image(panel, 'generated_images/crack/custom_crack.jpg')
```

---

### 2단계: Crack 감지

#### 기본 감지

```python
from defect_analyzer import DefectAnalyzer

# 분석기 생성
analyzer = DefectAnalyzer(
    min_defect_area=30,      # 최소 결함 크기 (Crack은 얇지만 길 수 있음)
    max_defect_area=20000,   # 최대 결함 크기
    threshold_method='adaptive'  # 이진화 방법
)

# 이미지 분석
image_path = 'generated_images/crack/my_crack_panel.jpg'
defects = analyzer.analyze_image(image_path)

# Crack만 필터링
cracks = [d for d in defects if d.defect_type == 'crack']
print(f"감지된 Crack: {len(cracks)}개")

# 각 Crack 정보 출력
for i, crack in enumerate(cracks, 1):
    print(f"\nCrack {i}:")
    print(f"  - 위치: ({crack.centroid[0]}, {crack.centroid[1]})")
    print(f"  - 면적: {crack.area:.2f} 픽셀")
    print(f"  - 둘레: {crack.perimeter:.2f} 픽셀")
    print(f"  - 종횡비: {crack.aspect_ratio:.3f}")
    print(f"  - 심각도: {crack.severity}")
```

#### 결과 시각화

```python
from pathlib import Path

results_dir = Path('test_results/crack')
results_dir.mkdir(parents=True, exist_ok=True)

# 결과 시각화 (Crack은 청록색으로 표시됨)
result_path = results_dir / 'crack_result.jpg'
analyzer.visualize_results(
    image_path,
    defects,
    save_path=str(result_path),
    show=True  # False로 설정하면 화면에 표시하지 않음
)

print(f"결과 이미지 저장: {result_path}")
```

#### 리포트 생성

```python
# CSV 리포트 생성
report_path = results_dir / 'crack_report.csv'
analyzer.generate_report(defects, str(report_path))
```

---

### 3단계: 완전한 예제

```python
from crack_generator import CrackImageGenerator
from defect_analyzer import DefectAnalyzer
from pathlib import Path
import math

# 폴더 설정
generated_dir = Path('generated_images/crack')
results_dir = Path('test_results/crack')
generated_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

print("=== Crack 이미지 생성 및 감지 ===\n")

# 1. Crack 이미지 생성
print("1. Crack 이미지 생성 중...")
generator = CrackImageGenerator(width=1920, height=1080)

# 여러 유형의 Crack 생성
panel = generator.generate_base_panel()

# 상단 가장자리에서 시작하는 직선 Crack
panel = generator.create_crack(
    panel, (200, 10), 300, math.pi / 2, 'straight', 2, 0.5
)

# 좌측 가장자리에서 시작하는 곡선 Crack
panel = generator.create_crack(
    panel, (10, 300), 250, 0.2, 'curved', 2, 0.6
)

# 하단 가장자리에서 시작하는 분기 Crack
panel = generator.create_crack(
    panel, (1500, 1070), 350, -math.pi / 2, 'branching', 3, 0.5
)

# 저장
image_path = generated_dir / 'sample_crack.jpg'
generator.save_image(panel, str(image_path))
print(f"   저장 완료: {image_path}\n")

# 2. Crack 감지
print("2. Crack 감지 중...")
analyzer = DefectAnalyzer(min_defect_area=30)
defects = analyzer.analyze_image(str(image_path))

# Crack만 필터링
cracks = [d for d in defects if d.defect_type == 'crack']
print(f"   총 감지된 결함: {len(defects)}개")
print(f"   Crack으로 분류: {len(cracks)}개\n")

# 3. 결과 시각화
print("3. 결과 시각화 중...")
result_path = results_dir / 'sample_crack_result.jpg'
analyzer.visualize_results(
    str(image_path),
    defects,
    save_path=str(result_path),
    show=False
)
print(f"   결과 저장: {result_path}\n")

# 4. 리포트 생성
print("4. 리포트 생성 중...")
report_path = results_dir / 'sample_crack_report.csv'
analyzer.generate_report(defects, str(report_path))
print(f"   리포트 저장: {report_path}\n")

print("=== 완료 ===")
```

---

## CLI를 통한 실행

### 단일 이미지 분석

```bash
python main.py generated_images/crack/test_straight_crack.jpg -o output/
```

### 배치 처리

```bash
python main.py generated_images/crack/ -o output/
```

---

## Crack 감지 특징

Crack 감지 알고리즘은 다음 특징을 기반으로 작동합니다:

1. **위치 기반**: 외곽(이미지 경계의 5% 이내)에서 시작하는 결함만 감지
2. **선형 구조**: 종횡비 > 8 또는 < 0.125인 선형 형태 감지
3. **형태 분석**: 높은 perimeter/area 비율로 선형 구조 식별
4. **크기 필터링**: 50~10,000 픽셀 범위의 결함만 Crack으로 분류

---

## 파라미터 조정

### 감지 민감도 조정

```python
analyzer = DefectAnalyzer(
    min_defect_area=20,      # 더 작은 Crack도 감지 (기본값: 30)
    max_defect_area=30000,   # 더 큰 Crack도 감지 (기본값: 20000)
    threshold_method='otsu'  # 이진화 방법 변경
)
```

### 이미지 생성 파라미터

```python
panel = generator.generate_panel_with_crack(
    num_cracks=3,                    # Crack 개수
    crack_types=['straight'],         # 유형
    min_length=100,                   # 최소 길이
    max_length=500                    # 최대 길이
)
```

---

## 문제 해결

### Crack이 감지되지 않는 경우

1. **min_defect_area 값 확인**: 너무 크면 작은 Crack을 놓칠 수 있음
2. **이미지 품질 확인**: 너무 어둡거나 밝으면 감지가 어려울 수 있음
3. **외곽 위치 확인**: Crack이 정말 외곽(5% 이내)에서 시작하는지 확인

### 너무 많은 결함이 감지되는 경우

1. **max_defect_area 값 감소**: 크기 범위를 좁힘
2. **threshold_method 변경**: 'otsu' 또는 'manual' 시도
3. **이미지 전처리**: 노이즈가 많은 경우 전처리 필요

---

## 출력 파일 위치

- **생성된 이미지**: `generated_images/crack/`
- **감지 결과**: `test_results/crack/`
- **일반 분석 결과**: `output/` (CLI 사용 시)

---

## 추가 참고

- 전체 프로젝트 문서: `README.md`
- 빠른 시작 가이드: `QUICKSTART.md`
- Chipping 감지 가이드: `test_chipping_detection.py` 참고

