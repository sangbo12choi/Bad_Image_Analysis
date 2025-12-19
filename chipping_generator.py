"""
Display Panel Chipping Image Generator
가상의 Chipping 불량이 있는 패널 이미지를 생성합니다.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import random


class ChippingImageGenerator:
    """Chipping 불량 이미지 생성기"""
    
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
    
    def create_chipping(self, 
                       image: np.ndarray,
                       position: Tuple[int, int],
                       size: Tuple[int, int],
                       chipping_type: str = 'corner',
                       depth: float = 0.3) -> np.ndarray:
        """
        Chipping 결함 생성
        
        Args:
            image: 대상 이미지
            position: Chipping 위치 (x, y) - 모서리나 가장자리 권장
            size: Chipping 크기 (width, height)
            chipping_type: Chipping 유형 ('corner', 'edge', 'irregular')
            depth: 깊이 (0~1, 클수록 깊게)
            
        Returns:
            Chipping이 추가된 이미지
        """
        result = image.copy()
        x, y = position
        w, h = size
        
        # Chipping 영역 마스크 생성
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        if chipping_type == 'corner':
            # 모서리 Chipping (불규칙한 삼각형/사각형 형태)
            points = self._generate_corner_chipping_shape(x, y, w, h, depth)
            cv2.fillPoly(mask, [points], 255)
            
        elif chipping_type == 'edge':
            # 가장자리 Chipping (불규칙한 직선 형태)
            points = self._generate_edge_chipping_shape(x, y, w, h, depth)
            cv2.fillPoly(mask, [points], 255)
            
        else:  # irregular
            # 불규칙한 형태의 Chipping
            points = self._generate_irregular_chipping_shape(x, y, w, h, depth)
            cv2.fillPoly(mask, [points], 255)
        
        # Chipping 영역에 어두운 색상 적용 (깨진 부분처럼)
        chipping_color = (int(50 * (1 - depth)), int(50 * (1 - depth)), int(50 * (1 - depth)))
        
        # 마스크 적용
        result[mask > 0] = chipping_color
        
        # Chipping 경계에 그림자 효과 추가
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        border = mask_dilated - mask
        
        # 경계를 더 어둡게
        darker_border = (int(30 * (1 - depth)), int(30 * (1 - depth)), int(30 * (1 - depth)))
        result[border > 0] = darker_border
        
        return result
    
    def _generate_corner_chipping_shape(self, x: int, y: int, w: int, h: int, 
                                       depth: float) -> np.ndarray:
        """모서리 Chipping 형태 생성"""
        # 불규칙한 모서리 형태 생성
        num_points = random.randint(5, 10)
        points = []
        
        # 시작점 (모서리)
        start_x, start_y = x, y
        
        # 불규칙한 경로 생성
        for i in range(num_points + 1):
            t = i / num_points
            # 베지어 곡선처럼 불규칙하게
            offset_x = random.uniform(-w * 0.2, w * 0.2) * (1 - t)
            offset_y = random.uniform(-h * 0.2, h * 0.2) * (1 - t)
            
            px = int(start_x + w * t + offset_x)
            py = int(start_y + h * t + offset_y)
            points.append([px, py])
        
        return np.array(points, dtype=np.int32)
    
    def _generate_edge_chipping_shape(self, x: int, y: int, w: int, h: int, 
                                     depth: float) -> np.ndarray:
        """가장자리 Chipping 형태 생성"""
        # 가장자리를 따라 불규칙한 형태
        num_points = random.randint(8, 15)
        points = []
        
        # 가장자리 방향 결정 (상/하/좌/우)
        edge_side = random.choice(['top', 'bottom', 'left', 'right'])
        
        if edge_side == 'top':
            for i in range(num_points + 1):
                px = int(x + w * (i / num_points))
                offset = random.uniform(-h * depth, h * depth * 0.5)
                py = int(y + offset)
                points.append([px, py])
            # 내부로 깊이 들어가는 점들
            points.append([x + w, y + int(h * depth)])
            points.append([x, y + int(h * depth)])
            
        elif edge_side == 'bottom':
            for i in range(num_points + 1):
                px = int(x + w * (i / num_points))
                offset = random.uniform(-h * depth * 0.5, h * depth)
                py = int(y + h - offset)
                points.append([px, py])
            points.append([x, y + h - int(h * depth)])
            points.append([x + w, y + h - int(h * depth)])
            
        elif edge_side == 'left':
            for i in range(num_points + 1):
                py = int(y + h * (i / num_points))
                offset = random.uniform(-w * depth, w * depth * 0.5)
                px = int(x + offset)
                points.append([px, py])
            points.append([x + int(w * depth), y])
            points.append([x + int(w * depth), y + h])
            
        else:  # right
            for i in range(num_points + 1):
                py = int(y + h * (i / num_points))
                offset = random.uniform(-w * depth * 0.5, w * depth)
                px = int(x + w - offset)
                points.append([px, py])
            points.append([x + w - int(w * depth), y + h])
            points.append([x + w - int(w * depth), y])
        
        return np.array(points, dtype=np.int32)
    
    def _generate_irregular_chipping_shape(self, x: int, y: int, w: int, h: int, 
                                          depth: float) -> np.ndarray:
        """불규칙한 Chipping 형태 생성"""
        # 완전히 불규칙한 형태
        num_points = random.randint(10, 20)
        center_x = x + w // 2
        center_y = y + h // 2
        
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            # 불규칙한 반지름
            radius_x = w / 2 * (0.5 + random.uniform(0, depth))
            radius_y = h / 2 * (0.5 + random.uniform(0, depth))
            
            px = int(center_x + radius_x * np.cos(angle) + random.uniform(-w*0.1, w*0.1))
            py = int(center_y + radius_y * np.sin(angle) + random.uniform(-h*0.1, h*0.1))
            points.append([px, py])
        
        return np.array(points, dtype=np.int32)
    
    def generate_panel_with_chipping(self,
                                    num_chippings: int = 1,
                                    chipping_types: Optional[list] = None,
                                    base_color: Tuple[int, int, int] = (240, 240, 240)) -> np.ndarray:
        """
        Chipping이 있는 패널 이미지 생성
        
        Args:
            num_chippings: 생성할 Chipping 개수
            chipping_types: Chipping 유형 리스트 (None이면 랜덤)
            base_color: 기본 패널 색상
            
        Returns:
            Chipping이 있는 패널 이미지
        """
        # 기본 패널 생성
        panel = self.generate_base_panel(base_color)
        
        if chipping_types is None:
            chipping_types = ['corner', 'edge', 'irregular']
        
        # Chipping 생성
        for i in range(num_chippings):
            chipping_type = random.choice(chipping_types)
            
            # 위치 결정 (모서리나 가장자리에 주로 발생)
            edge_choice = random.choice(['corner', 'edge', 'edge', 'edge'])  # 가장자리가 더 흔함
            
            if edge_choice == 'corner':
                # 모서리 선택
                corner = random.choice([
                    (0, 0),  # 좌상
                    (self.width - 200, 0),  # 우상
                    (0, self.height - 200),  # 좌하
                    (self.width - 200, self.height - 200)  # 우하
                ])
                size = (random.randint(50, 200), random.randint(50, 200))
                position = corner
            else:
                # 가장자리 선택
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    position = (random.randint(0, self.width - 200), 0)
                    size = (random.randint(100, 300), random.randint(30, 100))
                elif side == 'bottom':
                    position = (random.randint(0, self.width - 200), self.height - 100)
                    size = (random.randint(100, 300), random.randint(30, 100))
                elif side == 'left':
                    position = (0, random.randint(0, self.height - 200))
                    size = (random.randint(30, 100), random.randint(100, 300))
                else:  # right
                    position = (self.width - 100, random.randint(0, self.height - 200))
                    size = (random.randint(30, 100), random.randint(100, 300))
            
            depth = random.uniform(0.2, 0.6)
            
            panel = self.create_chipping(
                panel, position, size, chipping_type, depth
            )
        
        return panel
    
    def save_image(self, image: np.ndarray, filepath: str):
        """이미지 저장"""
        cv2.imwrite(filepath, image)
        print(f"이미지 저장 완료: {filepath}")


if __name__ == '__main__':
    # 예제: Chipping 이미지 생성
    from pathlib import Path
    
    # 출력 폴더 생성
    output_dir = Path('generated_images/chipping')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ChippingImageGenerator(width=1920, height=1080)
    
    # 단일 Chipping
    panel1 = generator.generate_panel_with_chipping(
        num_chippings=1,
        chipping_types=['corner']
    )
    generator.save_image(panel1, str(output_dir / 'sample_chipping_corner.jpg'))
    
    # 여러 Chipping
    panel2 = generator.generate_panel_with_chipping(
        num_chippings=3,
        chipping_types=['corner', 'edge', 'irregular']
    )
    generator.save_image(panel2, str(output_dir / 'sample_chipping_multiple.jpg'))

