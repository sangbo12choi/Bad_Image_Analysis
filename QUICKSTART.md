# 빠른 시작 가이드

## 1단계: 환경 설정

### 패키지 설치
```bash
pip install -r requirements.txt
```

## 2단계: 테스트 실행 (가장 간단한 방법)

### Chipping 감지 테스트
```bash
python test_chipping_detection.py
```

이 명령어를 실행하면:
- 가상의 Chipping 이미지가 자동으로 생성됩니다
- 각 이미지에서 Chipping이 감지됩니다
- 결과 이미지가 저장됩니다

생성되는 파일:
- `test_*.jpg`: 원본 Chipping 이미지
- `test_*_result.jpg`: 감지 결과 이미지
- `sample_chipping.jpg`: 샘플 이미지
- `sample_chipping_result.jpg`: 샘플 결과

## 3단계: 실제 이미지 분석

### 방법 A: 명령줄 사용 (간단)

```bash
# 단일 이미지 분석
python main.py your_image.jpg -o results

# 폴더 내 모든 이미지 분석
python main.py input_folder/ -o output_folder/
```

### 방법 B: Python 코드 사용 (유연함)

```python
from defect_analyzer import DefectAnalyzer

# 분석기 생성
analyzer = DefectAnalyzer()

# 이미지 분석
defects = analyzer.analyze_image('your_image.jpg')

# 결과 확인
print(f"감지된 결함: {len(defects)}개")
for defect in defects:
    print(f"- {defect.defect_type}: {defect.severity} (면적: {defect.area:.0f})")

# 결과 시각화
analyzer.visualize_results('your_image.jpg', defects, save_path='result.jpg')

# 리포트 생성
analyzer.generate_report(defects, 'report.csv')
```

## 4단계: Chipping 전용 분석

### 가상 Chipping 이미지 생성 및 감지

```python
from chipping_generator import ChippingImageGenerator
from defect_analyzer import DefectAnalyzer

# 1. Chipping 이미지 생성
generator = ChippingImageGenerator(width=1920, height=1080)
panel = generator.generate_panel_with_chipping(
    num_chippings=2,
    chipping_types=['corner', 'edge']
)
generator.save_image(panel, 'test_chipping.jpg')

# 2. Chipping 감지
analyzer = DefectAnalyzer(min_defect_area=50)
defects = analyzer.analyze_image('test_chipping.jpg')

# 3. Chipping만 필터링
chippings = [d for d in defects if d.defect_type == 'chipping']
print(f"감지된 Chipping: {len(chippings)}개")

# 4. 결과 시각화
analyzer.visualize_results('test_chipping.jpg', defects, save_path='result.jpg')
```

## 실행 예시

### 예시 1: 기본 분석
```bash
python main.py sample_image.jpg
```
→ `output/` 폴더에 결과 저장

### 예시 2: 민감도 조정
```bash
python main.py sample_image.jpg --min-area 20 --threshold otsu
```
→ 더 작은 결함도 감지

### 예시 3: 배치 처리
```bash
python main.py input_images/ -o results/ --no-show
```
→ 여러 이미지를 한 번에 처리하고 화면에는 표시하지 않음

## 문제 해결

### 패키지 설치 오류
```bash
# 개별 설치 시도
pip install opencv-python numpy scikit-image matplotlib pandas scipy Pillow
```

### 이미지를 찾을 수 없음
- 이미지 경로가 올바른지 확인
- 상대 경로 대신 절대 경로 사용 시도
- 이미지 파일 확장자 확인 (.jpg, .png 등)

### 결과가 표시되지 않음
- `--no-show` 옵션을 사용하지 않았는지 확인
- matplotlib 백엔드 문제일 수 있음 (GUI 환경 필요)

## 다음 단계

- `defect_analyzer.py`의 파라미터를 조정하여 감지 민감도 변경
- 실제 패널 이미지로 테스트하여 최적의 설정 찾기
- 배치 처리로 대량의 이미지 자동 분석

