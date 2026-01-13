"""
ChippingImageGenerator 단위 테스트
RED 단계: 실패하는 테스트 작성
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from chipping_generator import ChippingImageGenerator


class TestChippingImageGenerator:
    """ChippingImageGenerator 클래스 테스트"""
    
    def test_generator_initialization(self):
        """생성기 초기화 테스트"""
        generator = ChippingImageGenerator(width=1920, height=1080)
        assert generator.width == 1920
        assert generator.height == 1080
    
    def test_generate_base_panel(self):
        """기본 패널 생성 테스트"""
        generator = ChippingImageGenerator(width=100, height=100)
        
        panel = generator.generate_base_panel()
        
        assert panel is not None
        assert panel.shape == (100, 100, 3)  # BGR
        assert panel.dtype == np.uint8
    
    def test_generate_base_panel_custom_color(self):
        """커스텀 색상으로 패널 생성 테스트"""
        generator = ChippingImageGenerator(width=100, height=100)
        
        custom_color = (200, 200, 200)
        panel = generator.generate_base_panel(base_color=custom_color)
        
        assert panel is not None
        # 색상이 대략 맞는지 확인 (노이즈로 인해 정확히 같지 않을 수 있음)
        assert np.mean(panel) > 150  # 대략적인 색상 범위
    
    def test_create_chipping_corner(self):
        """모서리 Chipping 생성 테스트"""
        generator = ChippingImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_chipping(
            panel,
            position=(0, 0),
            size=(50, 50),
            chipping_type='corner',
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
        # Chipping이 추가되어 색상이 변경되었는지 확인
        assert not np.array_equal(result, panel)
    
    def test_create_chipping_edge(self):
        """가장자리 Chipping 생성 테스트"""
        generator = ChippingImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_chipping(
            panel,
            position=(0, 0),
            size=(100, 30),
            chipping_type='edge',
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
    
    def test_create_chipping_irregular(self):
        """불규칙한 Chipping 생성 테스트"""
        generator = ChippingImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_chipping(
            panel,
            position=(50, 50),
            size=(80, 80),
            chipping_type='irregular',
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
    
    def test_generate_panel_with_chipping(self):
        """Chipping이 있는 패널 생성 테스트"""
        generator = ChippingImageGenerator(width=200, height=200)
        
        panel = generator.generate_panel_with_chipping(
            num_chippings=2,
            chipping_types=['corner', 'edge']
        )
        
        assert panel is not None
        assert panel.shape == (200, 200, 3)
    
    def test_save_image(self):
        """이미지 저장 테스트"""
        generator = ChippingImageGenerator(width=100, height=100)
        panel = generator.generate_base_panel()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            generator.save_image(panel, tmp_path)
            
            assert os.path.exists(tmp_path), "이미지 파일이 생성되지 않았습니다"
            assert os.path.getsize(tmp_path) > 0, "이미지 파일이 비어있습니다"
            
            # 저장된 이미지가 올바른지 확인
            loaded = cv2.imread(tmp_path)
            assert loaded is not None
            assert loaded.shape == panel.shape
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_multiple_chippings(self):
        """여러 Chipping 생성 테스트"""
        generator = ChippingImageGenerator(width=500, height=500)
        
        panel = generator.generate_panel_with_chipping(
            num_chippings=5,
            chipping_types=['corner', 'edge', 'irregular']
        )
        
        assert panel is not None
        assert panel.shape == (500, 500, 3)
    
    def test_chipping_different_depths(self):
        """다양한 깊이의 Chipping 생성 테스트"""
        generator = ChippingImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        depths = [0.2, 0.5, 0.8]
        results = []
        
        for depth in depths:
            result = generator.create_chipping(
                panel.copy(),
                position=(10, 10),
                size=(30, 30),
                chipping_type='corner',
                depth=depth
            )
            results.append(result)
        
        # 깊이가 다르면 결과도 달라야 함
        assert len(results) == 3
        # 최소한 하나는 원본과 다르야 함
        assert any(not np.array_equal(r, panel) for r in results)

