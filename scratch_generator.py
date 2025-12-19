"""
Display Panel Scratch Image Generator
가상의 Scratch(스크래치/긁힘) 불량이 있는 패널 이미지를 생성합니다.
Scratch는 패널 표면 어디서나 발생할 수 있습니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import random
import math


class ScratchImageGenerator:
    """Scratch 불량 이미지 생성기"""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        """
        Args:
            width: 패널 이미지 너비
            height: 패널 이미지 높이
        """
        self.width = width
        self.height = height
    
    def generate_base_panel(self, 
                           base_color: Tuple[int, int, int] = (240, 240, 240),
                           noise_level: float = 0.02) -> np.ndarray:
        """
        기본 패널 이미지 생성 (약간의 노이즈 포함)
        
        Args:
            base_color: 기본 배경색 (BGR)
            noise_level: 노이즈 레벨 (0~1)
            
        Returns:
            기본 패널 이미지
        """
        # 기본 패널 생성
        panel = np.full((self.height, self.width, 3), base_color, dtype=np.uint8)
        
        # 약간의 노이즈 추가 (실제 패널처럼)
        noise = np.random.normal(0, noise_level * 255, panel.shape).astype(np.int16)
        panel = np.clip(panel.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # 가우시안 블러로 부드럽게
        panel = cv2.GaussianBlur(panel, (5, 5), 0)
        
        return panel
    
    def create_scratch(self, 
                      image: np.ndarray,
                      start_point: Tuple[int, int],
                      length: float,
                      direction: float,
                      scratch_type: str = 'straight',
                      width: int = 1,
                      depth: float = 0.4) -> np.ndarray:
        """
        Scratch 생성 (패널 표면 어디서나 발생 가능)
        
        Args:
            image: 대상 이미지
            start_point: Scratch 시작점 (x, y)
            length: Scratch 길이 (픽셀)
            direction: Scratch 방향 (라디안, 0=오른쪽, π/2=아래)
            scratch_type: Scratch 유형 ('straight', 'curved', 'zigzag')
            width: Scratch 두께 (픽셀, 보통 1-2)
            depth: 깊이/어둠 정도 (0~1)
            
        Returns:
            Scratch가 추가된 이미지
        """
        result = image.copy()
        
        # Scratch 경로 생성
        if scratch_type == 'straight':
            points = self._generate_straight_scratch(start_point, length, direction)
        elif scratch_type == 'curved':
            points = self._generate_curved_scratch(start_point, length, direction)
        else:  # zigzag
            points = self._generate_zigzag_scratch(start_point, length, direction)
        
        # Scratch 그리기
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i + 1]))
                
                # 어두운 색상 (Scratch)
                scratch_color = (
                    int(40 * (1 - depth)),
                    int(40 * (1 - depth)),
                    int(40 * (1 - depth))
                )
                
                # 얇은 선 그리기
                cv2.line(result, pt1, pt2, scratch_color, width)
                
                # 더 어두운 중심선 (깊이 효과)
                if width > 1:
                    darker_color = (
                        int(20 * (1 - depth)),
                        int(20 * (1 - depth)),
                        int(20 * (1 - depth))
                    )
                    cv2.line(result, pt1, pt2, darker_color, 1)
        
        return result
    
    def _generate_straight_scratch(self, start: Tuple[int, int], 
                                  length: float, direction: float) -> List[Tuple[float, float]]:
        """직선 Scratch 경로 생성"""
        points = []
        num_segments = max(10, int(length / 3))
        
        for i in range(num_segments + 1):
            t = i / num_segments
            # 약간의 미세한 불규칙성 추가 (실제 긁힘처럼)
            noise_angle = random.uniform(-0.05, 0.05) * (1 - t * 0.3)
            current_direction = direction + noise_angle
            
            x = start[0] + length * t * math.cos(current_direction)
            y = start[1] + length * t * math.sin(current_direction)
            
            # 이미지 경계 내로 제한
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            points.append((x, y))
        
        return points
    
    def _generate_curved_scratch(self, start: Tuple[int, int], 
                                length: float, direction: float) -> List[Tuple[float, float]]:
        """곡선 Scratch 경로 생성"""
        points = []
        num_segments = max(20, int(length / 2))
        
        # 베지어 곡선처럼 부드럽게
        control_points = [
            start,
            (
                start[0] + length * 0.3 * math.cos(direction + random.uniform(-0.3, 0.3)),
                start[1] + length * 0.3 * math.sin(direction + random.uniform(-0.3, 0.3))
            ),
            (
                start[0] + length * 0.7 * math.cos(direction + random.uniform(-0.2, 0.2)),
                start[1] + length * 0.7 * math.sin(direction + random.uniform(-0.2, 0.2))
            ),
            (
                start[0] + length * math.cos(direction + random.uniform(-0.1, 0.1)),
                start[1] + length * math.sin(direction + random.uniform(-0.1, 0.1))
            )
        ]
        
        for i in range(num_segments + 1):
            t = i / num_segments
            # 3차 베지어 곡선
            x = (1-t)**3 * control_points[0][0] + \
                3*(1-t)**2*t * control_points[1][0] + \
                3*(1-t)*t**2 * control_points[2][0] + \
                t**3 * control_points[3][0]
            
            y = (1-t)**3 * control_points[0][1] + \
                3*(1-t)**2*t * control_points[1][1] + \
                3*(1-t)*t**2 * control_points[2][1] + \
                t**3 * control_points[3][1]
            
            # 이미지 경계 내로 제한
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            points.append((x, y))
        
        return points
    
    def _generate_zigzag_scratch(self, start: Tuple[int, int], 
                               length: float, direction: float) -> List[Tuple[float, float]]:
        """지그재그 Scratch 경로 생성"""
        points = []
        num_segments = max(15, int(length / 4))
        zigzag_frequency = random.uniform(3, 8)  # 지그재그 빈도
        
        for i in range(num_segments + 1):
            t = i / num_segments
            
            # 지그재그 패턴
            zigzag_offset = math.sin(t * zigzag_frequency * math.pi) * (length * 0.1)
            perpendicular_angle = direction + math.pi / 2
            offset_x = zigzag_offset * math.cos(perpendicular_angle)
            offset_y = zigzag_offset * math.sin(perpendicular_angle)
            
            x = start[0] + length * t * math.cos(direction) + offset_x
            y = start[1] + length * t * math.sin(direction) + offset_y
            
            # 이미지 경계 내로 제한
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            points.append((x, y))
        
        return points
    
    def generate_panel_with_scratch(self,
                                   num_scratches: int = 1,
                                   scratch_types: Optional[List[str]] = None,
                                   base_color: Tuple[int, int, int] = (240, 240, 240),
                                   min_length: int = 100,
                                   max_length: int = 500) -> np.ndarray:
        """
        Scratch가 있는 패널 이미지 생성 (표면 어디서나 발생 가능)
        
        Args:
            num_scratches: 생성할 Scratch 개수
            scratch_types: Scratch 유형 리스트 (None이면 랜덤)
            base_color: 기본 패널 색상
            min_length: 최소 Scratch 길이
            max_length: 최대 Scratch 길이
            
        Returns:
            Scratch가 있는 패널 이미지
        """
        # 기본 패널 생성
        panel = self.generate_base_panel(base_color)
        
        if scratch_types is None:
            scratch_types = ['straight', 'curved', 'zigzag']
        
        # Scratch 생성 (패널 표면 어디서나 발생 가능)
        for i in range(num_scratches):
            scratch_type = random.choice(scratch_types)
            
            # 랜덤 시작점 (패널 표면 어디서나)
            start_x = random.randint(50, self.width - 50)
            start_y = random.randint(50, self.height - 50)
            start_point = (start_x, start_y)
            
            # 랜덤 방향
            direction = random.uniform(0, 2 * math.pi)
            
            length = random.uniform(min_length, max_length)
            width = random.randint(1, 2)  # Scratch는 보통 얇음 (1-2 픽셀)
            depth = random.uniform(0.3, 0.6)
            
            panel = self.create_scratch(
                panel, start_point, length, direction, scratch_type, width, depth
            )
        
        return panel
    
    def save_image(self, image: np.ndarray, filepath: str):
        """이미지 저장"""
        cv2.imwrite(filepath, image)
        print(f"이미지 저장 완료: {filepath}")


if __name__ == '__main__':
    # 예제: Scratch 이미지 생성
    from pathlib import Path
    
    # 출력 폴더 생성
    output_dir = Path('generated_images/scratch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ScratchImageGenerator(width=1920, height=1080)
    
    # 단일 Scratch
    panel1 = generator.generate_panel_with_scratch(
        num_scratches=1,
        scratch_types=['straight']
    )
    generator.save_image(panel1, str(output_dir / 'sample_scratch_straight.jpg'))
    
    # 여러 Scratch
    panel2 = generator.generate_panel_with_scratch(
        num_scratches=3,
        scratch_types=['straight', 'curved', 'zigzag']
    )
    generator.save_image(panel2, str(output_dir / 'sample_scratch_multiple.jpg'))

