"""
Display Panel Crack Image Generator
가상의 Crack(균열) 불량이 있는 패널 이미지를 생성합니다.
Crack은 Panel의 외곽(가장자리)에서만 발생합니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import random
import math


class CrackImageGenerator:
    """Crack 불량 이미지 생성기"""
    
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
    
    def create_crack(self, 
                    image: np.ndarray,
                    start_point: Tuple[int, int],
                    length: float,
                    direction: float,
                    crack_type: str = 'straight',
                    width: int = 2,
                    depth: float = 0.5) -> np.ndarray:
        """
        Crack 생성 (외곽에서 시작)
        
        Args:
            image: 대상 이미지
            start_point: Crack 시작점 (x, y) - 외곽에 위치해야 함
            length: Crack 길이 (픽셀)
            direction: Crack 방향 (라디안, 0=오른쪽, π/2=아래)
            crack_type: Crack 유형 ('straight', 'curved', 'branching')
            width: Crack 두께 (픽셀)
            depth: 깊이/어둠 정도 (0~1)
            
        Returns:
            Crack이 추가된 이미지
        """
        result = image.copy()
        
        # Crack 경로 생성
        if crack_type == 'straight':
            points = self._generate_straight_crack(start_point, length, direction)
        elif crack_type == 'curved':
            points = self._generate_curved_crack(start_point, length, direction)
        else:  # branching
            points = self._generate_branching_crack(start_point, length, direction)
        
        # Crack 그리기
        if len(points) > 1:
            # 다각형으로 두께 있는 선 그리기
            for i in range(len(points) - 1):
                pt1 = tuple(map(int, points[i]))
                pt2 = tuple(map(int, points[i + 1]))
                
                # 어두운 색상 (Crack)
                crack_color = (
                    int(30 * (1 - depth)),
                    int(30 * (1 - depth)),
                    int(30 * (1 - depth))
                )
                
                # 두께 있는 선 그리기
                cv2.line(result, pt1, pt2, crack_color, width)
                
                # 더 어두운 중심선 (깊이 효과)
                if width > 1:
                    darker_color = (
                        int(15 * (1 - depth)),
                        int(15 * (1 - depth)),
                        int(15 * (1 - depth))
                    )
                    cv2.line(result, pt1, pt2, darker_color, 1)
        
        return result
    
    def _generate_straight_crack(self, start: Tuple[int, int], 
                                length: float, direction: float) -> List[Tuple[float, float]]:
        """직선 Crack 경로 생성"""
        points = []
        num_segments = max(10, int(length / 5))
        
        for i in range(num_segments + 1):
            t = i / num_segments
            # 약간의 불규칙성 추가
            noise_angle = random.uniform(-0.1, 0.1) * (1 - t * 0.5)
            current_direction = direction + noise_angle
            
            x = start[0] + length * t * math.cos(current_direction)
            y = start[1] + length * t * math.sin(current_direction)
            
            # 이미지 경계 내로 제한
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            points.append((x, y))
        
        return points
    
    def _generate_curved_crack(self, start: Tuple[int, int], 
                              length: float, direction: float) -> List[Tuple[float, float]]:
        """곡선 Crack 경로 생성"""
        points = []
        num_segments = max(20, int(length / 3))
        
        # 베지어 곡선처럼 부드럽게
        control_points = [
            start,
            (
                start[0] + length * 0.3 * math.cos(direction + random.uniform(-0.5, 0.5)),
                start[1] + length * 0.3 * math.sin(direction + random.uniform(-0.5, 0.5))
            ),
            (
                start[0] + length * 0.7 * math.cos(direction + random.uniform(-0.3, 0.3)),
                start[1] + length * 0.7 * math.sin(direction + random.uniform(-0.3, 0.3))
            ),
            (
                start[0] + length * math.cos(direction + random.uniform(-0.2, 0.2)),
                start[1] + length * math.sin(direction + random.uniform(-0.2, 0.2))
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
    
    def _generate_branching_crack(self, start: Tuple[int, int], 
                                 length: float, direction: float) -> List[Tuple[float, float]]:
        """분기되는 Crack 경로 생성"""
        points = []
        num_segments = max(15, int(length / 4))
        
        # 메인 경로
        branch_point = int(num_segments * 0.6)
        
        for i in range(num_segments + 1):
            t = i / num_segments
            
            if i < branch_point:
                # 메인 경로
                noise_angle = random.uniform(-0.15, 0.15)
                current_direction = direction + noise_angle
            else:
                # 분기 경로
                branch_angle = random.uniform(-0.8, 0.8)
                current_direction = direction + branch_angle
            
            x = start[0] + length * t * math.cos(current_direction)
            y = start[1] + length * t * math.sin(current_direction)
            
            # 이미지 경계 내로 제한
            x = max(0, min(self.width - 1, x))
            y = max(0, min(self.height - 1, y))
            
            points.append((x, y))
        
        return points
    
    def generate_panel_with_crack(self,
                                  num_cracks: int = 1,
                                  crack_types: Optional[List[str]] = None,
                                  base_color: Tuple[int, int, int] = (240, 240, 240),
                                  min_length: int = 100,
                                  max_length: int = 500) -> np.ndarray:
        """
        Crack이 있는 패널 이미지 생성 (외곽에서만 시작)
        
        Args:
            num_cracks: 생성할 Crack 개수
            crack_types: Crack 유형 리스트 (None이면 랜덤)
            base_color: 기본 패널 색상
            min_length: 최소 Crack 길이
            max_length: 최대 Crack 길이
            
        Returns:
            Crack이 있는 패널 이미지
        """
        # 기본 패널 생성
        panel = self.generate_base_panel(base_color)
        
        if crack_types is None:
            crack_types = ['straight', 'curved', 'branching']
        
        # 외곽 경계 정의
        edge_threshold = 0.05  # 이미지 크기의 5% 이내가 외곽
        
        for i in range(num_cracks):
            crack_type = random.choice(crack_types)
            
            # 외곽에서 시작점 선택
            side = random.choice(['top', 'bottom', 'left', 'right'])
            
            if side == 'top':
                start_x = random.randint(0, self.width - 1)
                start_y = random.randint(0, int(self.height * edge_threshold))
                # 아래쪽으로 향하는 방향
                direction = random.uniform(math.pi / 2 - 0.5, math.pi / 2 + 0.5)
            elif side == 'bottom':
                start_x = random.randint(0, self.width - 1)
                start_y = random.randint(int(self.height * (1 - edge_threshold)), self.height - 1)
                # 위쪽으로 향하는 방향
                direction = random.uniform(-math.pi / 2 - 0.5, -math.pi / 2 + 0.5)
            elif side == 'left':
                start_x = random.randint(0, int(self.width * edge_threshold))
                start_y = random.randint(0, self.height - 1)
                # 오른쪽으로 향하는 방향
                direction = random.uniform(-0.5, 0.5)
            else:  # right
                start_x = random.randint(int(self.width * (1 - edge_threshold)), self.width - 1)
                start_y = random.randint(0, self.height - 1)
                # 왼쪽으로 향하는 방향
                direction = random.uniform(math.pi - 0.5, math.pi + 0.5)
            
            start_point = (start_x, start_y)
            length = random.uniform(min_length, max_length)
            width = random.randint(1, 3)  # Crack 두께
            depth = random.uniform(0.4, 0.7)
            
            panel = self.create_crack(
                panel, start_point, length, direction, crack_type, width, depth
            )
        
        return panel
    
    def save_image(self, image: np.ndarray, filepath: str):
        """이미지 저장"""
        cv2.imwrite(filepath, image)
        print(f"이미지 저장 완료: {filepath}")


if __name__ == '__main__':
    # 예제: Crack 이미지 생성
    from pathlib import Path
    
    # 출력 폴더 생성
    output_dir = Path('generated_images/crack')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = CrackImageGenerator(width=1920, height=1080)
    
    # 단일 Crack
    panel1 = generator.generate_panel_with_crack(
        num_cracks=1,
        crack_types=['straight']
    )
    generator.save_image(panel1, str(output_dir / 'sample_crack_straight.jpg'))
    
    # 여러 Crack
    panel2 = generator.generate_panel_with_crack(
        num_cracks=3,
        crack_types=['straight', 'curved', 'branching']
    )
    generator.save_image(panel2, str(output_dir / 'sample_crack_multiple.jpg'))

