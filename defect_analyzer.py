"""
Panel Hard Defect Analyzer
디스플레이 패널의 Hard Defect를 이미지 분석을 통해 감지하고 분류합니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
from skimage import morphology, measure, filters


@dataclass
class Defect:
    """결함 정보를 저장하는 데이터 클래스"""
    id: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    area: float
    centroid: Tuple[float, float]
    defect_type: str  # 'point', 'line', 'area', 'edge', 'chipping', 'crack', 'scratch'
    severity: str  # 'minor', 'moderate', 'severe'
    perimeter: float
    aspect_ratio: float
    solidity: float


class DefectAnalyzer:
    """패널 결함 분석 클래스"""
    
    def __init__(self, 
                 min_defect_area: int = 10,
                 max_defect_area: int = 100000,
                 threshold_method: str = 'adaptive',
                 morphology_kernel_size: int = 3):
        """
        Args:
            min_defect_area: 최소 결함 크기 (픽셀)
            max_defect_area: 최대 결함 크기 (픽셀)
            threshold_method: 이진화 방법 ('adaptive', 'otsu', 'manual')
            morphology_kernel_size: 형태학적 연산 커널 크기
        """
        self.min_defect_area = min_defect_area
        self.max_defect_area = max_defect_area
        self.threshold_method = threshold_method
        self.morphology_kernel_size = morphology_kernel_size
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지 전처리: 노이즈 제거 및 대비 향상
        
        Args:
            image: 입력 이미지 (BGR 또는 Grayscale)
            
        Returns:
            전처리된 그레이스케일 이미지
        """
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 가우시안 블러로 노이즈 제거
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def detect_defects(self, image: np.ndarray) -> np.ndarray:
        """
        결함 영역 감지
        
        Args:
            image: 전처리된 그레이스케일 이미지
            
        Returns:
            이진화된 결함 마스크
        """
        # 이진화
        if self.threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
        elif self.threshold_method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # 수동 임계값 (평균 기반)
            threshold_value = np.mean(image) * 0.7
            _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 형태학적 연산으로 노이즈 제거 및 결함 강화
        kernel = np.ones((self.morphology_kernel_size, self.morphology_kernel_size), np.uint8)
        
        # 작은 노이즈 제거 (Opening)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 결함 영역 확장 (Closing)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed
    
    def detect_chipping(self, image: np.ndarray) -> np.ndarray:
        """
        Chipping 결함 감지 (어두운 영역이 가장자리/모서리에 있는 경우)
        
        Args:
            image: 전처리된 그레이스케일 이미지
            
        Returns:
            Chipping 마스크
        """
        # Chipping은 어두운 영역이므로 낮은 임계값 사용
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 평균보다 훨씬 어두운 영역 감지 (Chipping)
        threshold_value = mean_intensity - 2 * std_intensity
        threshold_value = max(0, min(threshold_value, 100))  # 0~100 범위로 제한
        
        _, chipping_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 가장자리/모서리 영역 마스크 생성
        height, width = image.shape
        edge_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 가장자리 영역 (이미지 크기의 10% 이내)
        edge_threshold = 0.1
        edge_mask[:int(height * edge_threshold), :] = 255  # 상단
        edge_mask[int(height * (1 - edge_threshold)):, :] = 255  # 하단
        edge_mask[:, :int(width * edge_threshold)] = 255  # 좌측
        edge_mask[:, int(width * (1 - edge_threshold)):] = 255  # 우측
        
        # 모서리 영역 강조
        corner_size = min(width, height) * 0.15
        edge_mask[:int(corner_size), :int(corner_size)] = 255  # 좌상
        edge_mask[:int(corner_size), int(width - corner_size):] = 255  # 우상
        edge_mask[int(height - corner_size):, :int(corner_size)] = 255  # 좌하
        edge_mask[int(height - corner_size):, int(width - corner_size):] = 255  # 우하
        
        # Chipping은 가장자리/모서리에 있는 어두운 영역
        chipping_mask = cv2.bitwise_and(chipping_binary, edge_mask)
        
        # 형태학적 연산으로 정제
        kernel = np.ones((5, 5), np.uint8)
        chipping_mask = cv2.morphologyEx(chipping_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        chipping_mask = cv2.morphologyEx(chipping_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return chipping_mask
    
    def detect_crack(self, image: np.ndarray) -> np.ndarray:
        """
        Crack 결함 감지 (외곽에서 시작하는 선형 균열)
        
        Args:
            image: 전처리된 그레이스케일 이미지
            
        Returns:
            Crack 마스크
        """
        # Crack은 어두운 선형 구조이므로 낮은 임계값 사용
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 평균보다 훨씬 어두운 영역 감지
        threshold_value = mean_intensity - 1.5 * std_intensity
        threshold_value = max(0, min(threshold_value, 120))
        
        _, crack_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 외곽 영역 마스크 생성 (이미지 크기의 5% 이내)
        height, width = image.shape
        edge_threshold = 0.05
        edge_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 외곽 영역만 마스크
        edge_mask[:int(height * edge_threshold), :] = 255  # 상단
        edge_mask[int(height * (1 - edge_threshold)):, :] = 255  # 하단
        edge_mask[:, :int(width * edge_threshold)] = 255  # 좌측
        edge_mask[:, int(width * (1 - edge_threshold)):] = 255  # 우측
        
        # 외곽에서 시작하는 어두운 영역만
        crack_mask = cv2.bitwise_and(crack_binary, edge_mask)
        
        # 선형 구조 강화 (Crack은 선형이므로)
        # 형태학적 연산: 선형 구조 강조
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        kernel_diag1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # 다양한 방향의 선형 구조 감지
        horizontal = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel_horizontal)
        vertical = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel_vertical)
        diagonal = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel_diag1)
        
        # 결합
        crack_mask = cv2.bitwise_or(crack_mask, horizontal)
        crack_mask = cv2.bitwise_or(crack_mask, vertical)
        crack_mask = cv2.bitwise_or(crack_mask, diagonal)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return crack_mask
    
    def detect_scratch(self, image: np.ndarray) -> np.ndarray:
        """
        Scratch 결함 감지 (패널 표면 어디서나 발생하는 선형 긁힘)
        
        Args:
            image: 전처리된 그레이스케일 이미지
            
        Returns:
            Scratch 마스크
        """
        # Scratch는 어두운 선형 구조이므로 낮은 임계값 사용
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 평균보다 어두운 영역 감지
        threshold_value = mean_intensity - 1.5 * std_intensity
        threshold_value = max(0, min(threshold_value, 120))
        
        _, scratch_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 선형 구조 강화 (Scratch는 선형이므로)
        # 다양한 방향의 선형 구조 감지
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        kernel_diag1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        
        # 다양한 방향의 선형 구조 감지
        horizontal = cv2.morphologyEx(scratch_binary, cv2.MORPH_CLOSE, kernel_horizontal)
        vertical = cv2.morphologyEx(scratch_binary, cv2.MORPH_CLOSE, kernel_vertical)
        diagonal = cv2.morphologyEx(scratch_binary, cv2.MORPH_CLOSE, kernel_diag1)
        
        # 결합
        scratch_mask = cv2.bitwise_or(scratch_binary, horizontal)
        scratch_mask = cv2.bitwise_or(scratch_mask, vertical)
        scratch_mask = cv2.bitwise_or(scratch_mask, diagonal)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        scratch_mask = cv2.morphologyEx(scratch_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return scratch_mask
    
    def detect_bubble(self, image: np.ndarray) -> np.ndarray:
        """
        Bubble 결함 감지 (원형 또는 타원형 형태의 기포)
        
        Args:
            image: 전처리된 그레이스케일 이미지
            
        Returns:
            Bubble 마스크
        """
        # Bubble은 밝은 영역이거나 어두운 영역일 수 있음
        # 일반적으로 어두운 원형 구조로 나타남
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # 평균보다 어두운 영역 감지 (Bubble은 보통 어두움)
        threshold_value = mean_intensity - 1.0 * std_intensity
        threshold_value = max(0, min(threshold_value, 150))
        
        _, bubble_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 원형 구조 강화 (Bubble은 원형이므로)
        # 원형 커널 사용
        kernel_circular = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # 원형 구조 강조
        bubble_mask = cv2.morphologyEx(bubble_binary, cv2.MORPH_CLOSE, kernel_circular, iterations=2)
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        bubble_mask = cv2.morphologyEx(bubble_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return bubble_mask
    
    def classify_defect(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """
        결함을 분류 (크기, 형태, 위치 기반)
        
        Args:
            contour: 결함 윤곽선
            image_shape: 이미지 크기 (height, width)
            
        Returns:
            결함 분류 정보 딕셔너리
        """
        # 기본 속성 계산
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area == 0 or perimeter == 0:
            return None
        
        # 바운딩 박스
        x, y, w, h = cv2.boundingRect(contour)
        
        # 중심점
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # 종횡비
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Solidity (실제 면적 / Convex Hull 면적)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        # 결함 유형 분류
        defect_type = self._classify_defect_type(area, aspect_ratio, solidity, w, h)
        
        # 심각도 분류
        severity = self._classify_severity(area, image_shape)
        
        # 엣지 결함 여부 확인
        height, width = image_shape
        edge_threshold = 0.05  # 이미지 크기의 5% 이내면 엣지로 간주
        is_edge = (x < width * edge_threshold or 
                  x + w > width * (1 - edge_threshold) or
                  y < height * edge_threshold or 
                  y + h > height * (1 - edge_threshold))
        
        # Scratch 여부 확인 (패널 표면 어디서나 발생하는 선형 긁힘)
        is_scratch = self._is_scratch_defect(
            x, y, w, h, area, solidity, aspect_ratio, perimeter, width, height
        )
        
        # Crack 여부 확인 (외곽에서 시작하는 선형 균열)
        is_crack = self._is_crack_defect(
            x, y, w, h, area, solidity, aspect_ratio, perimeter, width, height
        )
        
        # Chipping 여부 확인 (가장자리/모서리에 있고, 불규칙한 형태)
        is_chipping = self._is_chipping_defect(
            x, y, w, h, area, solidity, aspect_ratio, width, height
        )
        
        # 불량 유형 분류 (Chipping, Crack, Scratch만)
        if is_scratch:
            defect_type = 'scratch'
        elif is_crack:
            defect_type = 'crack'
        elif is_chipping:
            defect_type = 'chipping'
        elif is_edge:
            defect_type = 'edge'
        
        return {
            'bbox': (x, y, w, h),
            'area': area,
            'centroid': (cx, cy),
            'defect_type': defect_type,
            'severity': severity,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'is_edge': is_edge
        }
    
    def _is_chipping_defect(self, x: int, y: int, w: int, h: int, 
                           area: float, solidity: float, aspect_ratio: float,
                           img_width: int, img_height: int) -> bool:
        """
        Chipping 결함인지 판단
        
        Chipping 특징:
        - 가장자리/모서리에 위치
        - 불규칙한 형태 (낮은 solidity)
        - 적당한 크기
        """
        # 가장자리/모서리 영역 확인
        edge_threshold = 0.1  # 이미지 크기의 10% 이내
        corner_threshold = 0.15  # 모서리 영역
        
        is_near_edge = (
            x < img_width * edge_threshold or 
            x + w > img_width * (1 - edge_threshold) or
            y < img_height * edge_threshold or 
            y + h > img_height * (1 - edge_threshold)
        )
        
        # 모서리 영역 확인
        is_corner = (
            (x < img_width * corner_threshold and y < img_height * corner_threshold) or
            (x + w > img_width * (1 - corner_threshold) and y < img_height * corner_threshold) or
            (x < img_width * corner_threshold and y + h > img_height * (1 - corner_threshold)) or
            (x + w > img_width * (1 - corner_threshold) and y + h > img_height * (1 - corner_threshold))
        )
        
        # Chipping 판단 조건
        # 1. 가장자리/모서리에 위치
        # 2. 불규칙한 형태 (solidity < 0.85)
        # 3. 적당한 크기 (100 ~ 50000 픽셀)
        # 4. 종횡비가 극단적이지 않음 (0.1 ~ 10)
        
        if not is_near_edge:
            return False
        
        if solidity > 0.85:  # 너무 규칙한 형태는 Chipping이 아님
            return False
        
        if area < 100 or area > 50000:  # 크기 범위
            return False
        
        if aspect_ratio < 0.1 or aspect_ratio > 10:  # 극단적인 종횡비는 제외
            return False
        
        return True
    
    def _is_crack_defect(self, x: int, y: int, w: int, h: int,
                         area: float, solidity: float, aspect_ratio: float,
                         perimeter: float, img_width: int, img_height: int) -> bool:
        """
        Crack 결함인지 판단
        
        Crack 특징:
        - 외곽(이미지 경계의 5% 이내)에서 시작
        - 선형적인 형태 (매우 높은 종횡비 또는 매우 낮은 종횡비)
        - 얇고 긴 형태
        - 높은 perimeter/area 비율 (선형 구조)
        """
        # 외곽 영역 확인 (이미지 경계의 5% 이내)
        edge_threshold = 0.05
        
        # 시작점이 외곽에 있는지 확인
        start_near_edge = (
            x < img_width * edge_threshold or
            x + w > img_width * (1 - edge_threshold) or
            y < img_height * edge_threshold or
            y + h > img_height * (1 - edge_threshold)
        )
        
        if not start_near_edge:
            return False
        
        # Crack 판단 조건
        # 1. 외곽에서 시작
        # 2. 선형적인 형태 (종횡비가 매우 높거나 낮음)
        # 3. 얇고 긴 형태 (높은 perimeter/area 비율)
        # 4. 적당한 크기
        
        # 선형 구조 확인 (종횡비가 매우 높거나 낮음)
        is_linear = aspect_ratio > 8 or aspect_ratio < 0.125
        
        if not is_linear:
            return False
        
        # 선형 구조 비율 확인 (perimeter/area가 높으면 선형)
        if area > 0:
            linearity_ratio = perimeter * perimeter / area  # 선형 구조일수록 높음
            # 선형 구조는 이 비율이 높음 (원형은 약 4π, 선형은 훨씬 높음)
            if linearity_ratio < 20:  # 너무 낮으면 선형이 아님
                return False
        else:
            return False
        
        # 크기 범위 확인
        if area < 50 or area > 10000:  # Crack은 보통 중간 크기
            return False
        
        # 너무 규칙한 형태는 제외 (Crack은 약간 불규칙함)
        if solidity > 0.95:  # 너무 규칙하면 Crack이 아님
            return False
        
        return True
    
    def _is_scratch_defect(self, x: int, y: int, w: int, h: int,
                           area: float, solidity: float, aspect_ratio: float,
                           perimeter: float, img_width: int, img_height: int) -> bool:
        """
        Scratch 결함인지 판단
        
        Scratch 특징:
        - 패널 표면 어디서나 발생 가능 (외곽 제한 없음)
        - 선형적인 형태 (매우 높은 종횡비 또는 매우 낮은 종횡비)
        - 얇고 긴 형태
        - 높은 perimeter/area 비율 (선형 구조)
        """
        # Scratch 판단 조건
        # 1. 선형적인 형태 (종횡비가 매우 높거나 낮음)
        # 2. 얇고 긴 형태 (높은 perimeter/area 비율)
        # 3. 적당한 크기
        
        # 선형 구조 확인 (종횡비가 매우 높거나 낮음)
        is_linear = aspect_ratio > 6 or aspect_ratio < 0.167
        
        if not is_linear:
            return False
        
        # 선형 구조 비율 확인 (perimeter/area가 높으면 선형)
        if area > 0:
            linearity_ratio = perimeter * perimeter / area  # 선형 구조일수록 높음
            # 선형 구조는 이 비율이 높음 (원형은 약 4π, 선형은 훨씬 높음)
            if linearity_ratio < 15:  # 너무 낮으면 선형이 아님
                return False
        else:
            return False
        
        # 크기 범위 확인 (Scratch는 보통 작고 얇음)
        if area < 30 or area > 5000:  # Scratch는 보통 작은 편
            return False
        
        # 너무 규칙한 형태는 제외 (Scratch는 약간 불규칙할 수 있음)
        if solidity > 0.98:  # 너무 규칙하면 Scratch가 아님
            return False
        
        # Crack과의 차별화: 외곽에서 시작하지 않으면 Scratch
        edge_threshold = 0.05
        start_near_edge = (
            x < img_width * edge_threshold or
            x + w > img_width * (1 - edge_threshold) or
            y < img_height * edge_threshold or
            y + h > img_height * (1 - edge_threshold)
        )
        
        # 외곽에서 시작하지 않으면 Scratch로 분류
        # (외곽에서 시작하면 Crack일 가능성이 높음)
        if not start_near_edge:
            return True
        
        # 외곽에서 시작하더라도, 너무 얇고 길면 Scratch일 수 있음
        # 하지만 우선순위는 Crack이므로 False 반환
        return False
    
    def _is_bubble_defect(self, x: int, y: int, w: int, h: int,
                          area: float, solidity: float, aspect_ratio: float,
                          img_width: int, img_height: int) -> bool:
        """
        Bubble 결함인지 판단
        
        Bubble 특징:
        - 원형 또는 타원형 형태 (높은 solidity, 적당한 종횡비)
        - 패널 표면 어디서나 발생 가능
        - 적당한 크기
        """
        # Bubble 판단 조건
        # 1. 원형 또는 타원형 형태 (높은 solidity)
        # 2. 적당한 종횡비 (원형에 가까움, 0.5 ~ 2.0)
        # 3. 적당한 크기
        
        # 원형 형태 확인 (높은 solidity)
        if solidity < 0.85:  # 너무 불규칙하면 Bubble이 아님
            return False
        
        # 종횡비 확인 (원형에 가까움)
        # 원형은 종횡비가 1에 가까움, 타원형은 0.5 ~ 2.0 범위
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        # 크기 범위 확인 (Bubble은 보통 중간 크기)
        if area < 50 or area > 20000:  # Bubble 크기 범위
            return False
        
        return True
    
    def _classify_defect_type(self, area: float, aspect_ratio: float, 
                              solidity: float, width: int, height: int) -> str:
        """결함 유형 분류"""
        # 선 결함: 긴 형태 (높은 종횡비 또는 낮은 종횡비)
        if aspect_ratio > 5 or aspect_ratio < 0.2:
            return 'line'
        
        # 면 결함: 큰 면적
        if area > 1000:
            return 'area'
        
        # 점 결함: 작은 면적
        if area < 100:
            return 'point'
        
        # 중간 크기는 형태에 따라 분류
        if solidity < 0.7:  # 불규칙한 형태
            return 'area'
        
        return 'point'
    
    def _classify_severity(self, area: float, image_shape: Tuple[int, int]) -> str:
        """심각도 분류"""
        total_area = image_shape[0] * image_shape[1]
        area_ratio = area / total_area
        
        if area_ratio > 0.01:  # 전체 면적의 1% 이상
            return 'severe'
        elif area_ratio > 0.001:  # 전체 면적의 0.1% 이상
            return 'moderate'
        else:
            return 'minor'
    
    def analyze_image(self, image_path: str) -> List[Defect]:
        """
        이미지를 분석하여 결함을 감지하고 분류
        
        Args:
            image_path: 분석할 이미지 경로
            
        Returns:
            감지된 결함 리스트
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 전처리
        processed = self.preprocess_image(image)
        
        # 일반 결함 감지
        defect_mask = self.detect_defects(processed)
        
        # Chipping 감지
        chipping_mask = self.detect_chipping(processed)
        
        # Crack 감지
        crack_mask = self.detect_crack(processed)
        
        # Scratch 감지
        scratch_mask = self.detect_scratch(processed)
        
        # 모든 마스크 결합
        combined_mask = cv2.bitwise_or(defect_mask, chipping_mask)
        combined_mask = cv2.bitwise_or(combined_mask, crack_mask)
        combined_mask = cv2.bitwise_or(combined_mask, scratch_mask)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 결함 분류
        defects = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # 크기 필터링
            if area < self.min_defect_area or area > self.max_defect_area:
                continue
            
            # 분류
            defect_info = self.classify_defect(contour, processed.shape)
            if defect_info is None:
                continue
            
            # Defect 객체 생성
            defect = Defect(
                id=idx,
                bbox=defect_info['bbox'],
                area=defect_info['area'],
                centroid=defect_info['centroid'],
                defect_type=defect_info['defect_type'],
                severity=defect_info['severity'],
                perimeter=defect_info['perimeter'],
                aspect_ratio=defect_info['aspect_ratio'],
                solidity=defect_info['solidity']
            )
            defects.append(defect)
        
        return defects
    
    def visualize_results(self, image_path: str, defects: List[Defect], 
                         save_path: Optional[str] = None, show: bool = True):
        """
        분석 결과를 시각화
        
        Args:
            image_path: 원본 이미지 경로
            defects: 감지된 결함 리스트
            save_path: 저장할 경로 (None이면 저장 안 함)
            show: 화면에 표시할지 여부
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Matplotlib으로 시각화
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # 결함 유형별 색상
        color_map = {
            'point': 'red',
            'line': 'blue',
            'area': 'yellow',
            'edge': 'green',
            'chipping': 'magenta',  # Chipping은 자홍색으로 표시
            'crack': 'cyan',  # Crack은 청록색으로 표시
            'scratch': 'orange'  # Scratch는 주황색으로 표시
        }
        
        # 각 결함 표시
        for defect in defects:
            x, y, w, h = defect.bbox
            color = color_map.get(defect.defect_type, 'red')
            
            # 바운딩 박스 그리기
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 중심점 표시
            ax.plot(defect.centroid[0], defect.centroid[1], 'o', 
                   color=color, markersize=5)
            
            # 라벨 표시
            label = f"{defect.defect_type}\n{defect.severity}\nArea: {defect.area:.0f}"
            ax.text(x, y - 5, label, color=color, fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_title(f'Detected Defects: {len(defects)}', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"결과 이미지 저장: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self, defects: List[Defect], output_path: str):
        """
        분석 결과 리포트 생성 (CSV)
        
        Args:
            defects: 감지된 결함 리스트
            output_path: 저장할 CSV 파일 경로
        """
        import pandas as pd
        
        data = []
        for defect in defects:
            data.append({
                'ID': defect.id,
                'Type': defect.defect_type,
                'Severity': defect.severity,
                'Area': defect.area,
                'Centroid_X': defect.centroid[0],
                'Centroid_Y': defect.centroid[1],
                'BBox_X': defect.bbox[0],
                'BBox_Y': defect.bbox[1],
                'BBox_Width': defect.bbox[2],
                'BBox_Height': defect.bbox[3],
                'Perimeter': defect.perimeter,
                'Aspect_Ratio': defect.aspect_ratio,
                'Solidity': defect.solidity
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"리포트 저장: {output_path}")
        
        # 요약 통계 출력
        print("\n=== 결함 분석 요약 ===")
        print(f"총 결함 수: {len(defects)}")
        print(f"\n유형별 분포:")
        print(df['Type'].value_counts())
        print(f"\n심각도별 분포:")
        print(df['Severity'].value_counts())
    
    def batch_analyze(self, input_folder: str, output_folder: str):
        """
        폴더 내 모든 이미지를 배치로 분석
        
        Args:
            input_folder: 입력 이미지 폴더
            output_folder: 결과 저장 폴더
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 지원하는 이미지 확장자
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"총 {len(image_files)}개의 이미지 발견")
        
        for img_file in image_files:
            print(f"\n처리 중: {img_file.name}")
            try:
                defects = self.analyze_image(str(img_file))
                
                # 결과 이미지 저장
                result_img_path = output_path / f"{img_file.stem}_result.jpg"
                self.visualize_results(str(img_file), defects, 
                                     save_path=str(result_img_path), show=False)
                
                # 리포트 저장
                report_path = output_path / f"{img_file.stem}_report.csv"
                self.generate_report(defects, str(report_path))
                
                print(f"  - 감지된 결함: {len(defects)}개")
                
            except Exception as e:
                print(f"  - 오류 발생: {e}")

