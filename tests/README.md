# 테스트 가이드

## 테스트 구조

```
tests/
├── __init__.py
├── test_defect_analyzer.py      # DefectAnalyzer 단위 테스트
├── test_chipping_generator.py   # ChippingImageGenerator 테스트
├── test_crack_generator.py      # CrackImageGenerator 테스트
├── test_scratch_generator.py    # ScratchImageGenerator 테스트
├── test_integration.py          # 통합 테스트
└── test_new_features.py         # 새로운 기능 테스트 (RED 단계)
```

## 테스트 실행

### 모든 테스트 실행
```bash
pytest
```

### 특정 테스트 파일 실행
```bash
pytest tests/test_defect_analyzer.py
```

### 특정 테스트 실행
```bash
pytest tests/test_defect_analyzer.py::TestDefectAnalyzer::test_analyzer_initialization
```

### 커버리지 측정
```bash
pytest --cov=defect_analyzer --cov-report=html
```

### 실패한 테스트만 다시 실행
```bash
pytest --lf
```

## RED 단계 결과

현재 **5개의 테스트가 실패**하고 있습니다 (의도된 실패):

1. **test_crack_classification** - 윤곽선이 너무 작아서 분류되지 않음
2. **test_scratch_classification** - 윤곽선이 너무 작아서 분류되지 않음
3. **test_bubble_detection_exists** - 아직 구현되지 않은 Bubble 감지 기능
4. **test_bubble_detection_returns_mask** - 아직 구현되지 않은 Bubble 감지 기능
5. **test_bubble_classification** - 아직 구현되지 않은 Bubble 분류 기능

## 다음 단계 (GREEN)

RED 단계 완료 후:
1. 실패한 테스트를 통과시키는 최소한의 코드 작성
2. 테스트 실행하여 통과 확인
3. REFACTOR 단계로 코드 개선

## 테스트 작성 가이드

### AAA 패턴 사용
- **Arrange**: 테스트 준비
- **Act**: 테스트 실행
- **Assert**: 결과 검증

### 좋은 테스트의 특징
- 명확한 이름
- 독립적 (다른 테스트에 의존하지 않음)
- 반복 가능
- 빠름
- 명확한 실패 메시지

