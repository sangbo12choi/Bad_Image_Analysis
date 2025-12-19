# 코드 분석 리포트
## 정적 분석, 코드 스멜, SOLID 원칙 분석

**분석 일시**: 2024년
**분석 대상 파일**:
- `analyze_gui.py` (1,312 라인)
- `defect_analyzer.py` (819 라인)

---

## 1. 정적 분석 (Static Analysis)

### 1.1 파일 크기 및 복잡도

#### analyze_gui.py
- **총 라인 수**: 1,312 라인
- **클래스 수**: 1개 (DefectAnalysisGUI)
- **메서드 수**: 약 30개
- **평균 메서드 길이**: 약 43 라인
- **최대 메서드 길이**: 
  - `display_image()`: ~110 라인
  - `reset_application()`: ~86 라인
  - `create_highlighted_image()`: ~54 라인

#### defect_analyzer.py
- **총 라인 수**: 819 라인
- **클래스 수**: 1개 (DefectAnalyzer)
- **메서드 수**: 약 20개
- **평균 메서드 길이**: 약 40 라인
- **최대 메서드 길이**:
  - `classify_defect()`: ~87 라인
  - `analyze_image()`: ~67 라인

### 1.2 순환 복잡도 (Cyclomatic Complexity)

**높은 복잡도 메서드**:
1. `classify_defect()` (defect_analyzer.py:290) - 복잡도: ~15
   - 다중 if-elif 체인
   - 여러 불량 유형 판단 로직
   
2. `display_image()` (analyze_gui.py:893) - 복잡도: ~12
   - 다중 조건 분기
   - KOREAN_FONT_PROP 체크 반복
   
3. `_is_crack_defect()` (defect_analyzer.py:428) - 복잡도: ~10
   - 복잡한 조건문 중첩

### 1.3 중복 코드 (Code Duplication)

#### 심각한 중복:
1. **KOREAN_FONT_PROP 체크 반복** (analyze_gui.py)
   - `display_image()`: 4회 반복
   - `display_original_image()`: 2회 반복
   - `reset_application()`: 4회 반복
   - 총 약 10회 이상 반복
   
2. **색상 맵 정의 중복**
   - `display_image()`: color_map 정의 (라인 920-928)
   - `create_highlighted_image()`: color_map_rgb 정의 (라인 1031-1039)
   - `visualize_results()`: color_map 정의 (defect_analyzer.py:695-703)
   
3. **이미지 초기화 텍스트 중복**
   - "원본 이미지\n(이미지를 선택하세요)" - 3회 반복
   - "분석 결과\n(분석을 실행하세요)" - 3회 반복

4. **불량 유형 필터링 로직 중복**
   - `analyze_single()`: 라인 725
   - `analyze_batch()`: 라인 772
   - 동일한 필터링 로직이 두 곳에 존재

### 1.4 사용하지 않는 코드

1. **defect_analyzer.py**:
   - `detect_bubble()` 메서드 (라인 256-288): 정의되어 있지만 사용되지 않음
   - `_is_bubble_defect()` 메서드 (라인 543-572): 정의되어 있지만 사용되지 않음
   - `batch_analyze()` 메서드: GUI에서 사용되지 않음

2. **analyze_gui.py**:
   - `sys` import (라인 20): 사용되지 않음
   - `Image, ImageTk` from PIL (라인 18): 사용되지 않음

---

## 2. 코드 스멜 (Code Smells)

### 2.1 긴 메서드 (Long Method)

**문제가 있는 메서드들**:

1. **`display_image()`** (analyze_gui.py:893-1002, 110라인)
   - 책임: 이미지 표시, 불량 그리기, 라벨 표시, 폰트 처리
   - **문제**: 너무 많은 책임을 가짐
   - **권장**: 이미지 렌더링, 불량 그리기, 라벨 표시로 분리

2. **`reset_application()`** (analyze_gui.py:1210-1295, 86라인)
   - 책임: 모든 상태 초기화, UI 초기화
   - **문제**: 초기화 로직이 한 곳에 집중
   - **권장**: 상태 초기화, UI 초기화로 분리

3. **`classify_defect()`** (defect_analyzer.py:290-376, 87라인)
   - 책임: 결함 속성 계산, 유형 분류, 심각도 분류
   - **문제**: 복잡한 분류 로직이 한 메서드에 집중
   - **권장**: 속성 계산, 유형 분류, 심각도 분류로 분리

### 2.2 큰 클래스 (Large Class)

**DefectAnalysisGUI 클래스** (analyze_gui.py:99-1299)
- **라인 수**: 1,200+ 라인
- **메서드 수**: 약 30개
- **책임**: 
  - UI 생성 및 관리
  - 이미지 로드 및 표시
  - 분석 실행 및 결과 표시
  - 파일 저장
  - 상태 관리
- **문제**: 단일 책임 원칙 위반, 너무 많은 책임
- **권장**: 
  - UI 컴포넌트별로 분리 (ImageDisplay, StatisticsDisplay, ReportDisplay)
  - 비즈니스 로직 분리 (AnalysisController)

**DefectAnalyzer 클래스** (defect_analyzer.py:31-818)
- **라인 수**: 787 라인
- **메서드 수**: 약 20개
- **책임**:
  - 이미지 전처리
  - 불량 감지 (여러 유형)
  - 불량 분류
  - 결과 시각화
  - 리포트 생성
- **문제**: 단일 책임 원칙 위반
- **권장**:
  - 전처리기 분리 (ImagePreprocessor)
  - 감지기 분리 (DefectDetector)
  - 분류기 분리 (DefectClassifier)
  - 시각화 분리 (ResultVisualizer)

### 2.3 중복 코드 (Duplicated Code)

1. **폰트 처리 중복** (analyze_gui.py)
   - KOREAN_FONT_PROP 체크가 10회 이상 반복
   - **해결책**: 헬퍼 메서드 생성
   ```python
   def _add_text_with_font(self, ax, x, y, text, **kwargs):
       if KOREAN_FONT_PROP:
           ax.text(x, y, text, fontproperties=KOREAN_FONT_PROP, **kwargs)
       else:
           ax.text(x, y, text, **kwargs)
   ```

2. **색상 맵 중복**
   - 3곳에서 색상 맵 정의
   - **해결책**: 상수로 정의하여 재사용

3. **이미지 초기화 텍스트 중복**
   - **해결책**: 상수로 정의

### 2.4 긴 매개변수 목록 (Long Parameter List)

**문제가 있는 메서드들**:

1. **`_is_chipping_defect()`** (defect_analyzer.py:378)
   - 매개변수: 8개 (x, y, w, h, area, solidity, aspect_ratio, img_width, img_height)
   - **해결책**: DefectProperties 데이터 클래스 사용

2. **`_is_crack_defect()`** (defect_analyzer.py:428)
   - 매개변수: 9개
   - **해결책**: DefectProperties 데이터 클래스 사용

3. **`_is_scratch_defect()`** (defect_analyzer.py:485)
   - 매개변수: 9개
   - **해결책**: DefectProperties 데이터 클래스 사용

### 2.5 데이터 덩어리 (Data Clumps)

- **불량 속성들**: x, y, w, h, area, solidity, aspect_ratio 등이 여러 메서드에서 함께 전달됨
- **해결책**: DefectProperties 데이터 클래스 생성

### 2.6 기본 타입에 대한 집착 (Primitive Obsession)

- **불량 유형**: 문자열로 관리 ('chipping', 'crack', 'scratch')
- **심각도**: 문자열로 관리 ('minor', 'moderate', 'severe')
- **해결책**: Enum 클래스 사용

### 2.7 긴 조건문 (Long Conditional)

**문제가 있는 부분**:

1. **`classify_defect()`** (defect_analyzer.py:356-364)
   ```python
   if is_scratch:
       defect_type = 'scratch'
   elif is_crack:
       defect_type = 'crack'
   elif is_chipping:
       defect_type = 'chipping'
   elif is_edge:
       defect_type = 'edge'
   ```
   - **해결책**: 전략 패턴 또는 팩토리 패턴 사용

2. **이미지 확장자 체크** (analyze_gui.py:751-755)
   - 반복적인 glob 패턴
   - **해결책**: 헬퍼 메서드로 추출

### 2.8 주석 처리된 코드

- 발견되지 않음 (좋음)

### 2.9 매직 넘버 (Magic Numbers)

**문제가 있는 부분**:

1. **임계값들** (defect_analyzer.py)
   - `edge_threshold = 0.1` (여러 곳)
   - `corner_threshold = 0.15`
   - `threshold_value = mean_intensity - 2 * std_intensity`
   - **해결책**: 상수로 정의

2. **크기 범위들**
   - `area < 100 or area > 50000` (chipping)
   - `area < 50 or area > 10000` (crack)
   - **해결책**: 상수로 정의

---

## 3. SOLID 원칙 분석

### 3.1 Single Responsibility Principle (SRP) - 단일 책임 원칙

#### 위반 사례:

1. **DefectAnalysisGUI 클래스**
   - **책임 1**: UI 생성 및 관리
   - **책임 2**: 이미지 로드 및 표시
   - **책임 3**: 분석 실행 및 결과 처리
   - **책임 4**: 파일 저장
   - **책임 5**: 상태 관리
   - **위반 정도**: 심각 (5개 이상의 책임)

2. **DefectAnalyzer 클래스**
   - **책임 1**: 이미지 전처리
   - **책임 2**: 불량 감지
   - **책임 3**: 불량 분류
   - **책임 4**: 결과 시각화
   - **책임 5**: 리포트 생성
   - **위반 정도**: 심각 (5개 이상의 책임)

3. **`display_image()` 메서드**
   - 이미지 렌더링 + 불량 그리기 + 라벨 표시 + 폰트 처리
   - **위반 정도**: 중간

### 3.2 Open/Closed Principle (OCP) - 개방-폐쇄 원칙

#### 위반 사례:

1. **불량 유형 추가 시**
   - `classify_defect()` 메서드 수정 필요
   - `display_image()` 메서드 수정 필요 (color_map 추가)
   - `create_highlighted_image()` 메서드 수정 필요
   - **위반 정도**: 심각

2. **이진화 방법 추가 시**
   - `detect_defects()` 메서드 수정 필요
   - **위반 정도**: 중간

**개선 방안**:
- 전략 패턴 사용 (DefectDetectionStrategy, ThresholdStrategy)
- 팩토리 패턴 사용 (DefectClassifierFactory)

### 3.3 Liskov Substitution Principle (LSP) - 리스코프 치환 원칙

#### 현재 상태:
- 상속 구조가 없어 LSP 위반 사례 없음
- **평가**: 해당 없음

### 3.4 Interface Segregation Principle (ISP) - 인터페이스 분리 원칙

#### 현재 상태:
- 명시적인 인터페이스가 없음
- **평가**: 해당 없음 (Python의 덕 타이핑 특성상)

### 3.5 Dependency Inversion Principle (DIP) - 의존성 역전 원칙

#### 위반 사례:

1. **DefectAnalysisGUI → DefectAnalyzer**
   - 구체 클래스에 직접 의존
   - **위반 정도**: 중간
   - **개선 방안**: 추상 인터페이스 도입 (IDefectAnalyzer)

2. **하드코딩된 의존성**
   - `self.analyzer = DefectAnalyzer()` (analyze_gui.py:109)
   - **위반 정도**: 중간
   - **개선 방안**: 의존성 주입 (Dependency Injection)

---

## 4. 추가 발견 사항

### 4.1 에러 처리

**문제점**:
1. **일관성 없는 에러 처리**
   - 일부는 try-except 사용, 일부는 그냥 예외 발생
   - 사용자 친화적인 에러 메시지 부족

2. **로깅 부재**
   - print 문으로 디버깅 (프로덕션에 부적합)
   - **해결책**: logging 모듈 사용

### 4.2 테스트 가능성

**문제점**:
1. **GUI와 비즈니스 로직 결합**
   - GUI 코드에서 직접 분석 로직 호출
   - 단위 테스트 어려움

2. **전역 상태 의존**
   - KOREAN_FONT_PROP 전역 변수
   - 테스트 격리 어려움

### 4.3 성능 이슈

**잠재적 문제**:
1. **이미지 처리**
   - 매번 이미지를 다시 로드 (캐싱 부족)
   - 큰 이미지 처리 시 메모리 이슈 가능

2. **UI 업데이트**
   - `root.after()` 사용은 좋지만, 너무 많은 콜백 가능

---

## 5. 우선순위별 개선 사항

### 높은 우선순위 (High Priority)

1. **중복 코드 제거**
   - KOREAN_FONT_PROP 체크 헬퍼 메서드 생성
   - 색상 맵 상수화
   - 이미지 초기화 텍스트 상수화

2. **사용하지 않는 코드 제거**
   - `detect_bubble()`, `_is_bubble_defect()` 제거
   - 사용하지 않는 import 제거

3. **긴 메서드 분리**
   - `display_image()` 분리
   - `classify_defect()` 분리

### 중간 우선순위 (Medium Priority)

1. **SRP 위반 해결**
   - DefectAnalysisGUI 클래스 분리
   - DefectAnalyzer 클래스 분리

2. **OCP 위반 해결**
   - 전략 패턴 도입
   - 팩토리 패턴 도입

3. **매직 넘버 상수화**
   - 임계값 상수 정의
   - 크기 범위 상수 정의

### 낮은 우선순위 (Low Priority)

1. **DIP 개선**
   - 인터페이스 도입
   - 의존성 주입 도입

2. **로깅 시스템 도입**
   - print 문을 logging으로 교체

3. **테스트 가능성 개선**
   - 비즈니스 로직과 GUI 분리
   - 전역 상태 제거

---

## 6. 결론

### 전체 평가

**코드 품질 점수**: 6/10

**강점**:
- 기능은 잘 작동함
- 타입 힌트 사용 (일부)
- docstring 존재

**약점**:
- 단일 책임 원칙 위반 (심각)
- 중복 코드 다수
- 긴 메서드와 큰 클래스
- 테스트 어려움

**권장 사항**:
1. 즉시: 중복 코드 제거 및 사용하지 않는 코드 제거
2. 단기: 긴 메서드 분리 및 SRP 개선
3. 중장기: 아키텍처 리팩토링 (전략 패턴, 의존성 주입)

---

**리포트 작성 완료**

