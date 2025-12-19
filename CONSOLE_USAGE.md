# 불량별 이미지 분석 콘솔 프로그램 사용 가이드

## 개요

`analyze_console.py`는 불량 유형별로 이미지를 분석할 수 있는 인터랙티브 콘솔 프로그램입니다.

## 실행 방법

### 방법 1: 인터랙티브 모드 (권장)

인자 없이 실행하면 메뉴 기반 인터랙티브 모드로 실행됩니다.

```bash
python analyze_console.py
```

#### 실행 화면 예시

```
======================================================================
               불량별 이미지 분석 시스템
======================================================================

분석기 초기화 중...
✓ 분석기 초기화 완료

[메인 메뉴]
----------------------------------------------------------------------
1. 단일 이미지 분석
2. 배치 이미지 분석 (폴더)
3. 분석 설정 변경
4. 종료
----------------------------------------------------------------------
메뉴를 선택하세요 (1-4): 
```

### 방법 2: 명령줄 모드

기존 `main.py`와 동일하게 명령줄에서 직접 실행할 수 있습니다.

#### 단일 이미지 분석

```bash
python analyze_console.py 이미지경로.jpg -o 출력폴더
```

예시:
```bash
python analyze_console.py panel_image.jpg -o results
```

#### 특정 불량 유형만 분석

```bash
python analyze_console.py 이미지경로.jpg -o 출력폴더 --type chipping
```

사용 가능한 불량 유형:
- `all`: 전체 불량 (기본값)
- `chipping`: Chipping만
- `crack`: Crack만
- `scratch`: Scratch만
- `bubble`: Bubble만

예시:
```bash
# Chipping만 분석
python analyze_console.py panel.jpg -o results --type chipping

# Crack만 분석
python analyze_console.py panel.jpg -o results --type crack
```

#### 배치 처리 (폴더 내 모든 이미지)

```bash
python analyze_console.py 입력폴더/ -o 출력폴더/
```

예시:
```bash
python analyze_console.py input_images/ -o output_results/
```

#### 고급 옵션

```bash
python analyze_console.py 이미지.jpg -o 결과/ \
    --min-area 50 \
    --max-area 50000 \
    --threshold otsu \
    --type chipping \
    --no-show
```

옵션 설명:
- `-o, --output`: 결과 저장 폴더 (기본값: output)
- `--min-area`: 최소 결함 크기 (픽셀, 기본값: 10)
- `--max-area`: 최대 결함 크기 (픽셀, 기본값: 100000)
- `--threshold`: 이진화 방법 (adaptive/otsu/manual, 기본값: adaptive)
- `--type`: 분석할 불량 유형 (all/chipping/crack/scratch/bubble, 기본값: all)
- `--no-show`: 결과 이미지를 화면에 표시하지 않음

## 인터랙티브 모드 사용법

### 1. 단일 이미지 분석

1. 메뉴에서 `1` 선택
2. 이미지 파일 경로 입력
3. 불량 유형 선택:
   - `1`: 전체 불량
   - `2`: Chipping만
   - `3`: Crack만
   - `4`: Scratch만
   - `5`: Bubble만
   - `6`: 사용자 정의 필터
4. 결과 저장 폴더 입력 (기본: output)

#### 결과

- 분석 결과가 콘솔에 출력됩니다
- 결과 이미지가 저장됩니다 (`파일명_result.jpg`)
- CSV 리포트가 생성됩니다 (`파일명_report.csv`)

### 2. 배치 이미지 분석

1. 메뉴에서 `2` 선택
2. 입력 폴더 경로 입력
3. 불량 유형 선택
4. 결과 저장 폴더 입력

#### 결과

- 폴더 내 모든 이미지가 순차적으로 분석됩니다
- 각 이미지별로 결과 이미지와 리포트가 생성됩니다
- 전체 통계가 출력됩니다

### 3. 분석 설정 변경

1. 메뉴에서 `3` 선택
2. 최소 결함 크기 입력
3. 최대 결함 크기 입력
4. 이진화 방법 선택:
   - `1`: adaptive (적응형)
   - `2`: otsu (오츠)
   - `3`: manual (수동)

## 사용 예시

### 예시 1: Chipping만 분석

```bash
python analyze_console.py
```

메뉴에서:
1. `1` (단일 이미지 분석) 선택
2. 이미지 경로 입력: `generated_images/chipping/sample_chipping.jpg`
3. 불량 유형: `2` (Chipping) 선택
4. 출력 폴더: `results` 입력

### 예시 2: 배치 처리로 Crack 분석

```bash
python analyze_console.py
```

메뉴에서:
1. `2` (배치 이미지 분석) 선택
2. 입력 폴더: `generated_images/crack/` 입력
3. 불량 유형: `3` (Crack) 선택
4. 출력 폴더: `crack_results` 입력

### 예시 3: 명령줄에서 직접 실행

```bash
# Chipping만 분석
python analyze_console.py generated_images/chipping/sample_chipping.jpg -o results --type chipping

# 배치 처리 (전체 불량)
python analyze_console.py generated_images/ -o batch_results/

# 고급 설정으로 분석
python analyze_console.py image.jpg -o results --min-area 100 --max-area 10000 --threshold otsu --type scratch
```

## 출력 파일

### 결과 이미지 (`*_result.jpg`)

- 원본 이미지에 감지된 불량이 색상으로 표시됩니다
- 불량 유형별 색상:
  - Chipping: 자홍색 (magenta)
  - Crack: 청록색 (cyan)
  - Scratch: 주황색 (orange)
  - Bubble: 보라색 (purple)
  - 기타: 빨간색, 파란색, 노란색, 초록색

### 리포트 파일 (`*_report.csv`)

CSV 형식으로 다음 정보가 포함됩니다:
- ID: 결함 ID
- Type: 결함 유형
- Severity: 심각도 (minor/moderate/severe)
- Area: 면적 (픽셀)
- Centroid_X, Centroid_Y: 중심점 좌표
- BBox_X, BBox_Y, BBox_Width, BBox_Height: 바운딩 박스
- Perimeter: 둘레
- Aspect_Ratio: 종횡비
- Solidity: 견고성

## 팁

1. **불량 유형별 분석**: 특정 불량 유형만 분석하려면 `--type` 옵션을 사용하세요
2. **배치 처리**: 여러 이미지를 한 번에 분석하려면 폴더 경로를 입력하세요
3. **설정 조정**: 작은 불량을 감지하려면 `--min-area` 값을 낮추세요
4. **성능**: 배치 처리 시 `--no-show` 옵션을 사용하면 더 빠릅니다

## 문제 해결

### 이미지를 찾을 수 없습니다
- 파일 경로가 정확한지 확인하세요
- 상대 경로 또는 절대 경로를 사용하세요

### 불량이 감지되지 않습니다
- `--min-area` 값을 낮춰보세요
- `--threshold` 방법을 변경해보세요 (otsu, manual)
- 이미지 품질을 확인하세요

### 프로그램이 느립니다
- `--no-show` 옵션을 사용하세요
- 배치 처리 시 한 번에 너무 많은 이미지를 처리하지 마세요

## 기존 main.py와의 차이점

- ✅ 인터랙티브 메뉴 제공
- ✅ 불량 유형별 선택 가능
- ✅ 사용자 정의 필터 지원
- ✅ 설정 변경 기능
- ✅ 기존 명령줄 모드도 지원 (하위 호환)

`main.py`와 동일하게 사용할 수 있으며, 추가로 인터랙티브 모드와 불량 유형 필터링 기능이 추가되었습니다.

