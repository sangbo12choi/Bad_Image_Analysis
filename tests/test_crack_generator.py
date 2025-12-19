"""
CrackImageGenerator 단위 테스트
RED 단계: 실패하는 테스트 작성
"""

import pytest
import numpy as np
import cv2
import math
from pathlib import Path
import tempfile
import os

from crack_generator import CrackImageGenerator


class TestCrackImageGenerator:
    """CrackImageGenerator 클래스 테스트"""
    
    def test_generator_initialization(self):
        """생성기 초기화 테스트"""
        generator = CrackImageGenerator(width=1920, height=1080)
        assert generator.width == 1920
        assert generator.height == 1080
    
    def test_generate_base_panel(self):
        """기본 패널 생성 테스트"""
        generator = CrackImageGenerator(width=100, height=100)
        
        panel = generator.generate_base_panel()
        
        assert panel is not None
        assert panel.shape == (100, 100, 3)  # BGR
        assert panel.dtype == np.uint8
    
    def test_create_crack_straight(self):
        """직선 Crack 생성 테스트"""
        generator = CrackImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_crack(
            panel,
            start_point=(10, 10),  # 외곽에서 시작
            length=100,
            direction=math.pi / 2,  # 아래 방향
            crack_type='straight',
            width=2,
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
        # Crack이 추가되어 색상이 변경되었는지 확인
        assert not np.array_equal(result, panel)
    
    def test_create_crack_curved(self):
        """곡선 Crack 생성 테스트"""
        generator = CrackImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_crack(
            panel,
            start_point=(10, 10),
            length=100,
            direction=math.pi / 2,
            crack_type='curved',
            width=2,
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
    
    def test_create_crack_branching(self):
        """분기 Crack 생성 테스트"""
        generator = CrackImageGenerator(width=200, height=200)
        panel = generator.generate_base_panel()
        
        result = generator.create_crack(
            panel,
            start_point=(10, 10),
            length=100,
            direction=math.pi / 2,
            crack_type='branching',
            width=2,
            depth=0.5
        )
        
        assert result is not None
        assert result.shape == panel.shape
    
    def test_generate_panel_with_crack(self):
        """Crack이 있는 패널 생성 테스트"""
        generator = CrackImageGenerator(width=200, height=200)
        
        panel = generator.generate_panel_with_crack(
            num_cracks=2,
            crack_types=['straight', 'curved']
        )
        
        assert panel is not None
        assert panel.shape == (200, 200, 3)
    
    def test_crack_starts_from_edge(self):
        """Crack이 외곽에서 시작하는지 테스트"""
        generator = CrackImageGenerator(width=1000, height=1000)
        
        panel = generator.generate_panel_with_crack(
            num_cracks=4,  # 각 가장자리에서 하나씩
            crack_types=['straight']
        )
        
        assert panel is not None
        
        # RED: 실제로 외곽에서 시작하는지 확인하는 더 정교한 테스트 필요
        # 현재는 생성만 확인
    
    def test_save_image(self):
        """이미지 저장 테스트"""
        generator = CrackImageGenerator(width=100, height=100)
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
    
    def test_multiple_cracks(self):
        """여러 Crack 생성 테스트"""
        generator = CrackImageGenerator(width=500, height=500)
        
        panel = generator.generate_panel_with_crack(
            num_cracks=5,
            crack_types=['straight', 'curved', 'branching']
        )
        
        assert panel is not None
        assert panel.shape == (500, 500, 3)
    
    def test_crack_different_lengths(self):
        """다양한 길이의 Crack 생성 테스트"""
        generator = CrackImageGenerator(width=500, height=500)
        panel = generator.generate_base_panel()
        
        lengths = [100, 200, 300]
        results = []
        
        for length in lengths:
            result = generator.create_crack(
                panel.copy(),
                start_point=(10, 10),
                length=length,
                direction=math.pi / 2,
                crack_type='straight',
                width=2,
                depth=0.5
            )
            results.append(result)
        
        assert len(results) == 3
        # 길이가 다르면 결과도 달라야 함
        assert any(not np.array_equal(r, panel) for r in results)

