"""
ScratchImageGenerator 단위 테스트
RED 단계: 실패하는 테스트 작성
"""

import pytest
import numpy as np
import cv2
import math
from pathlib import Path
import tempfile
import os

from scratch_generator import ScratchImageGenerator


class TestScratchImageGenerator:
    """ScratchImageGenerator 클래스 테스트"""
    
    def test_generator_initialization(self):
        """생성기 초기화 테스트"""
        generator = ScratchImageGenerator(width=1920, height=1080)
        assert generator.width == 1920
        assert generator.height == 1080
    
    def test_generate_base_panel(self):
        """기본 패널 생성 테스트"""
        generator = ScratchImageGenerator(width=100, height=100)
        
        panel = generator.generate_base_panel()
        
        assert panel is not None
        assert panel.shape == (100, 100, 3)  # BGR
        assert panel.dtype == np.uint8
    
    def test_create_scratch_straight(self):
        """직선 Scratch 생성 테스트"""
        generator = ScratchImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_scratch(
            panel,
            start_point=(100, 100),  # 중앙에서 시작 (외곽 제한 없음)
            length=100,
            direction=math.pi / 4,  # 대각선 방향
            scratch_type='straight',
            width=1,
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
        # Scratch가 추가되어 색상이 변경되었는지 확인
        assert not np.array_equal(result, panel)
    
    def test_create_scratch_curved(self):
        """곡선 Scratch 생성 테스트"""
        generator = ScratchImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_scratch(
            panel,
            start_point=(100, 100),
            length=100,
            direction=math.pi / 4,
            scratch_type='curved',
            width=1,
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
    
    def test_create_scratch_zigzag(self):
        """지그재그 Scratch 생성 테스트"""
        generator = ScratchImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_scratch(
            panel,
            start_point=(100, 100),
            length=100,
            direction=math.pi / 4,
            scratch_type='zigzag',
            width=2,
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
    
    def test_generate_panel_with_scratch(self):
        """Scratch가 있는 패널 생성 테스트"""
        generator = ScratchImageGenerator(width=200, height=200)
        
        panel = generator.generate_panel_with_scratch(
            num_scratches=2,
            scratch_types=['straight', 'curved']
        )
        
        assert panel is not None
        assert panel.shape == (200, 200, 3)
    
    def test_scratch_can_be_anywhere(self):
        """Scratch가 패널 어디서나 생성되는지 테스트"""
        generator = ScratchImageGenerator(width=1000, height=1000)
        
        # 중앙, 모서리, 가장자리 등 다양한 위치에 Scratch 생성
        panel = generator.generate_base_panel()
        
        positions = [
            (500, 500),  # 중앙
            (100, 100),  # 좌상
            (900, 100),  # 우상
            (100, 900),  # 좌하
            (900, 900),  # 우하
        ]
        
        for pos in positions:
            result = generator.create_scratch(
                panel.copy(),
                start_point=pos,
                length=50,
                direction=math.pi / 4,
                scratch_type='straight',
                width=1,
                depth=0.5
            )
            assert result is not None
    
    def test_save_image(self):
        """이미지 저장 테스트"""
        generator = ScratchImageGenerator(width=100, height=100)
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
    
    def test_multiple_scratches(self):
        """여러 Scratch 생성 테스트"""
        generator = ScratchImageGenerator(width=500, height=500)
        
        panel = generator.generate_panel_with_scratch(
            num_scratches=5,
            scratch_types=['straight', 'curved', 'zigzag']
        )
        
        assert panel is not None
        assert panel.shape == (500, 500, 3)
    
    def test_scratch_different_widths(self):
        """다양한 두께의 Scratch 생성 테스트"""
        generator = ScratchImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        widths = [1, 2, 3]
        results = []
        
        for width in widths:
            result = generator.create_scratch(
                panel.copy(),
                start_point=(100, 100),
                length=50,
                direction=math.pi / 4,
                scratch_type='straight',
                width=width,
                depth=0.5
            )
            results.append(result)
        
        assert len(results) == 3
        # 두께가 다르면 결과도 달라야 함
        assert any(not np.array_equal(r, panel) for r in results)

