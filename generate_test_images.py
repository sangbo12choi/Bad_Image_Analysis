"""
불량 분석 테스트를 위한 가상 이미지 일괄 생성 스크립트
Chipping, Crack, Scratch 각각 20개씩 생성합니다.
"""

import os
from pathlib import Path
from chipping_generator import ChippingImageGenerator
from crack_generator import CrackImageGenerator
from scratch_generator import ScratchImageGenerator
import random
import math

def ensure_directory(path: Path):
    """디렉토리가 없으면 생성"""
    path.mkdir(parents=True, exist_ok=True)

def generate_chipping_images(output_dir: Path, count: int = 20):
    """Chipping 이미지 생성"""
    print(f"\n=== Chipping 이미지 {count}개 생성 중 ===")
    generator = ChippingImageGenerator(width=1920, height=1080)
    
    chipping_types = ['corner', 'edge', 'irregular']
    
    for i in range(count):
        # Chipping 개수 (1-3개)
        num_chippings = random.randint(1, 3)
        
        # Chipping 타입 리스트 생성
        selected_types = [random.choice(chipping_types) for _ in range(num_chippings)]
        
        # 패널 생성
        panel = generator.generate_panel_with_chipping(
            num_chippings=num_chippings,
            chipping_types=selected_types
        )
        
        # 파일 저장
        chipping_type_str = "_".join(selected_types)
        filename = f"chipping_{i+1:02d}_{chipping_type_str}.jpg"
        filepath = output_dir / filename
        generator.save_image(panel, str(filepath))
        print(f"  [OK] {filename} 생성 완료")

def generate_crack_images(output_dir: Path, count: int = 20):
    """Crack 이미지 생성"""
    print(f"\n=== Crack 이미지 {count}개 생성 중 ===")
    generator = CrackImageGenerator(width=1920, height=1080)
    
    crack_types = ['straight', 'curved', 'branching']
    
    for i in range(count):
        # Crack 개수 (1-2개)
        num_cracks = random.randint(1, 2)
        
        # Crack 타입 리스트 생성
        selected_types = [random.choice(crack_types) for _ in range(num_cracks)]
        
        # 패널 생성
        panel = generator.generate_panel_with_crack(
            num_cracks=num_cracks,
            crack_types=selected_types
        )
        
        # 파일 저장
        crack_type_str = "_".join(selected_types)
        filename = f"crack_{i+1:02d}_{crack_type_str}.jpg"
        filepath = output_dir / filename
        generator.save_image(panel, str(filepath))
        print(f"  [OK] {filename} 생성 완료")

def generate_scratch_images(output_dir: Path, count: int = 20):
    """Scratch 이미지 생성"""
    print(f"\n=== Scratch 이미지 {count}개 생성 중 ===")
    generator = ScratchImageGenerator(width=1920, height=1080)
    
    scratch_types = ['straight', 'curved', 'zigzag']
    
    for i in range(count):
        # Scratch 개수 (1-3개)
        num_scratches = random.randint(1, 3)
        
        # Scratch 타입 리스트 생성
        selected_types = [random.choice(scratch_types) for _ in range(num_scratches)]
        
        # 패널 생성
        panel = generator.generate_panel_with_scratch(
            num_scratches=num_scratches,
            scratch_types=selected_types
        )
        
        # 파일 저장
        scratch_type_str = "_".join(selected_types)
        filename = f"scratch_{i+1:02d}_{scratch_type_str}.jpg"
        filepath = output_dir / filename
        generator.save_image(panel, str(filepath))
        print(f"  [OK] {filename} 생성 완료")

def main():
    """메인 함수"""
    # 출력 디렉토리 설정
    base_dir = Path("generated_images")
    chipping_dir = base_dir / "chipping" / "test"
    crack_dir = base_dir / "crack" / "test"
    scratch_dir = base_dir / "scratch" / "test"
    
    # 디렉토리 생성
    ensure_directory(chipping_dir)
    ensure_directory(crack_dir)
    ensure_directory(scratch_dir)
    
    print("=" * 60)
    print("불량 분석 테스트용 가상 이미지 생성")
    print("=" * 60)
    
    # 이미지 생성
    try:
        generate_chipping_images(chipping_dir, count=20)
        generate_crack_images(crack_dir, count=20)
        generate_scratch_images(scratch_dir, count=20)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 모든 이미지 생성 완료!")
        print("=" * 60)
        print(f"\n생성된 이미지 위치:")
        print(f"  - Chipping: {chipping_dir}")
        print(f"  - Crack: {crack_dir}")
        print(f"  - Scratch: {scratch_dir}")
        print(f"\n총 생성된 이미지: 60개 (각 20개씩)")
        
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

