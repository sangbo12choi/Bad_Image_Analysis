# Panel Hard Defect Analysis System

디스플레이 패널의 Hard Defect 불량을 이미지 분석을 통해 자동으로 감지하고 분류하는 시스템입니다.

## 주요 기능

- **이미지 전처리**: 노이즈 제거, 대비 향상, 이진화
- **결함 감지**: 다양한 알고리즘을 통한 결함 영역 탐지
- **결함 분류**: 크기, 형태, 위치 기반 분류
- **결과 시각화**: 감지된 결함을 시각적으로 표시
- **리포트 생성**: 분석 결과를 CSV/이미지로 저장

## 설치 방법

```bash
pip install -r requirements.txt
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

## 파라미터 조정

`defect_analyzer.py`의 설정값을 조정하여 감지 민감도를 변경할 수 있습니다.

