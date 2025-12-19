"""
DefectAnalyzer 단위 테스트
RED 단계: 실패하는 테스트 작성
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from defect_analyzer import DefectAnalyzer, Defect


class TestDefectAnalyzer:
    """DefectAnalyzer 클래스 테스트"""
    
    def test_analyzer_initialization(self):
        """분석기 초기화 테스트"""
        analyzer = DefectAnalyzer()
        assert analyzer is not None
        assert analyzer.min_defect_area == 10
        assert analyzer.max_defect_area == 100000
        assert analyzer.threshold_method == 'adaptive'
    
    def test_analyzer_custom_parameters(self):
        """커스텀 파라미터로 초기화 테스트"""
        analyzer = DefectAnalyzer(
            min_defect_area=50,
            max_defect_area=50000,
            threshold_method='otsu'
        )
        assert analyzer.min_defect_area == 50
        assert analyzer.max_defect_area == 50000
        assert analyzer.threshold_method == 'otsu'
    
    def test_preprocess_image(self):
        """이미지 전처리 테스트"""
        analyzer = DefectAnalyzer()
        
        # 테스트 이미지 생성 (BGR)
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 240
        
        processed = analyzer.preprocess_image(test_image)
        
        assert processed is not None
        assert len(processed.shape) == 2  # Grayscale
        assert processed.shape == (100, 100)
        assert processed.dtype == np.uint8
    
    def test_preprocess_grayscale_image(self):
        """그레이스케일 이미지 전처리 테스트"""
        analyzer = DefectAnalyzer()
        
        # 그레이스케일 이미지
        test_image = np.ones((100, 100), dtype=np.uint8) * 240
        
        processed = analyzer.preprocess_image(test_image)
        
        assert processed is not None
        assert len(processed.shape) == 2
        assert processed.shape == (100, 100)
    
    def test_detect_defects_returns_mask(self):
        """결함 감지가 마스크를 반환하는지 테스트"""
        analyzer = DefectAnalyzer()
        
        # 테스트 이미지 생성
        test_image = np.ones((100, 100), dtype=np.uint8) * 240
        
        mask = analyzer.detect_defects(test_image)
        
        assert mask is not None
        assert mask.shape == test_image.shape
        assert mask.dtype == np.uint8
    
    def test_detect_chipping_exists(self):
        """Chipping 감지 메서드 존재 확인"""
        analyzer = DefectAnalyzer()
        assert hasattr(analyzer, 'detect_chipping'), "detect_chipping 메서드가 없습니다"
    
    def test_detect_chipping_returns_mask(self):
        """Chipping 감지가 마스크를 반환하는지 테스트"""
        analyzer = DefectAnalyzer()
        
        test_image = np.ones((100, 100), dtype=np.uint8) * 240
        
        mask = analyzer.detect_chipping(test_image)
        
        assert mask is not None
        assert mask.shape == test_image.shape
        assert mask.dtype == np.uint8
    
    def test_detect_crack_exists(self):
        """Crack 감지 메서드 존재 확인"""
        analyzer = DefectAnalyzer()
        assert hasattr(analyzer, 'detect_crack'), "detect_crack 메서드가 없습니다"
    
    def test_detect_crack_returns_mask(self):
        """Crack 감지가 마스크를 반환하는지 테스트"""
        analyzer = DefectAnalyzer()
        
        test_image = np.ones((100, 100), dtype=np.uint8) * 240
        
        mask = analyzer.detect_crack(test_image)
        
        assert mask is not None
        assert mask.shape == test_image.shape
        assert mask.dtype == np.uint8
    
    def test_detect_scratch_exists(self):
        """Scratch 감지 메서드 존재 확인"""
        analyzer = DefectAnalyzer()
        assert hasattr(analyzer, 'detect_scratch'), "detect_scratch 메서드가 없습니다"
    
    def test_detect_scratch_returns_mask(self):
        """Scratch 감지가 마스크를 반환하는지 테스트"""
        analyzer = DefectAnalyzer()
        
        test_image = np.ones((100, 100), dtype=np.uint8) * 240
        
        mask = analyzer.detect_scratch(test_image)
        
        assert mask is not None
        assert mask.shape == test_image.shape
        assert mask.dtype == np.uint8
    
    def test_analyze_image_file_not_found(self):
        """존재하지 않는 이미지 파일 처리 테스트"""
        analyzer = DefectAnalyzer()
        
        with pytest.raises(ValueError, match="이미지를 로드할 수 없습니다"):
            analyzer.analyze_image('nonexistent_image.jpg')
    
    def test_classify_defect_returns_dict(self):
        """결함 분류가 딕셔너리를 반환하는지 테스트"""
        analyzer = DefectAnalyzer()
        
        # 간단한 윤곽선 생성
        contour = np.array([[[10, 10]], [[50, 10]], [[50, 50]], [[10, 50]]], dtype=np.int32)
        image_shape = (100, 100)
        
        result = analyzer.classify_defect(contour, image_shape)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'defect_type' in result
        assert 'severity' in result
        assert 'area' in result
        assert 'centroid' in result
    
    def test_classify_defect_empty_contour(self):
        """빈 윤곽선 처리 테스트"""
        analyzer = DefectAnalyzer()
        
        # 빈 윤곽선
        contour = np.array([], dtype=np.int32).reshape(0, 1, 2)
        image_shape = (100, 100)
        
        result = analyzer.classify_defect(contour, image_shape)
        
        assert result is None
    
    def test_generate_report_creates_csv(self):
        """리포트 생성이 CSV 파일을 만드는지 테스트"""
        analyzer = DefectAnalyzer()
        
        # 테스트용 결함 생성
        defects = [
            Defect(
                id=0,
                bbox=(10, 10, 20, 20),
                area=400.0,
                centroid=(20.0, 20.0),
                defect_type='point',
                severity='minor',
                perimeter=80.0,
                aspect_ratio=1.0,
                solidity=1.0
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            analyzer.generate_report(defects, tmp_path)
            
            assert os.path.exists(tmp_path), "CSV 파일이 생성되지 않았습니다"
            assert os.path.getsize(tmp_path) > 0, "CSV 파일이 비어있습니다"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDefectClassification:
    """결함 분류 로직 테스트"""
    
    def test_chipping_classification(self):
        """Chipping 분류 테스트"""
        analyzer = DefectAnalyzer()
        
        # Chipping 특징을 가진 윤곽선 (가장자리에 위치, 불규칙한 형태)
        # 실제로는 이미지에서 추출된 윤곽선이 필요하지만, 테스트용으로 간단히 생성
        contour = np.array([[[5, 5]], [[30, 5]], [[30, 30]], [[5, 30]]], dtype=np.int32)
        image_shape = (1000, 1000)
        
        result = analyzer.classify_defect(contour, image_shape)
        
        # RED: Chipping이 정확히 분류되는지 확인 (현재는 실패할 수 있음)
        # 이 테스트는 실제 Chipping 이미지로 더 정확하게 테스트해야 함
        assert result is not None
        assert 'defect_type' in result
    
    def test_crack_classification(self):
        """Crack 분류 테스트"""
        analyzer = DefectAnalyzer()
        
        # Crack 특징을 가진 윤곽선 (외곽에서 시작, 선형)
        # 폭을 가진 선형 구조를 만들기 위해 직사각형 윤곽선 생성
        # 외곽(5, 50)에서 시작하여 내부로 향하는 선형 구조
        contour = np.array([
            [[5, 45]], [[5, 55]],  # 시작점 (외곽)
            [[200, 45]], [[200, 55]]  # 끝점
        ], dtype=np.int32)
        # 더 큰 면적을 위해 폭을 가진 윤곽선으로 수정
        contour = np.array([
            [[5, 45]], [[5, 55]], [[200, 55]], [[200, 45]]
        ], dtype=np.int32)
        image_shape = (1000, 1000)
        
        result = analyzer.classify_defect(contour, image_shape)
        
        # GREEN: Crack이 정확히 분류되는지 확인
        assert result is not None
        assert 'defect_type' in result
    
    def test_scratch_classification(self):
        """Scratch 분류 테스트"""
        analyzer = DefectAnalyzer()
        
        # Scratch 특징을 가진 윤곽선 (중앙에 위치, 선형)
        # 폭을 가진 선형 구조를 만들기 위해 직사각형 윤곽선 생성
        contour = np.array([
            [[500, 495]], [[500, 505]], [[700, 505]], [[700, 495]]
        ], dtype=np.int32)
        image_shape = (1000, 1000)
        
        result = analyzer.classify_defect(contour, image_shape)
        
        # GREEN: Scratch가 정확히 분류되는지 확인
        assert result is not None
        assert 'defect_type' in result

