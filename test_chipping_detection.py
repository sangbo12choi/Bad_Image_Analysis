"""
Chipping 감지 테스트 스크립트
가상의 Chipping 이미지를 생성하고 감지 기능을 테스트합니다.
"""

from chipping_generator import ChippingImageGenerator
from defect_analyzer import DefectAnalyzer
from pathlib import Path
import cv2
import numpy as np


def test_chipping_detection():
    """Chipping 감지 테스트"""
    print("=== Chipping 감지 테스트 ===\n")
    
    # 폴더 설정
    generated_dir = Path('generated_images/chipping')
    results_dir = Path('test_results/chipping')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Chipping 이미지 생성
    print("1. 가상 Chipping 이미지 생성 중...")
    generator = ChippingImageGenerator(width=1920, height=1080)
    
    # 다양한 유형의 Chipping 생성
    test_cases = [
        {
            'name': 'corner_chipping',
            'num_chippings': 1,
            'types': ['corner'],
            'description': '모서리 Chipping'
        },
        {
            'name': 'edge_chipping',
            'num_chippings': 1,
            'types': ['edge'],
            'description': '가장자리 Chipping'
        },
        {
            'name': 'irregular_chipping',
            'num_chippings': 1,
            'types': ['irregular'],
            'description': '불규칙한 Chipping'
        },
        {
            'name': 'multiple_chippings',
            'num_chippings': 3,
            'types': ['corner', 'edge', 'irregular'],
            'description': '다중 Chipping'
        }
    ]
    
    generated_images = []
    
    for test_case in test_cases:
        print(f"   - {test_case['description']} 생성 중...")
        panel = generator.generate_panel_with_chipping(
            num_chippings=test_case['num_chippings'],
            chipping_types=test_case['types']
        )
        
        filename = generated_dir / f"test_{test_case['name']}.jpg"
        generator.save_image(panel, str(filename))
        generated_images.append({
            'filename': str(filename),
            'description': test_case['description']
        })
    
    print("\n2. Chipping 감지 테스트 시작...\n")
    
    # 2. Chipping 감지
    analyzer = DefectAnalyzer(
        min_defect_area=50,  # Chipping은 보통 큰 편이므로
        max_defect_area=100000,
        threshold_method='adaptive'
    )
    
    for img_info in generated_images:
        print(f"--- {img_info['description']} 분석 ---")
        print(f"이미지: {img_info['filename']}")
        
        try:
            # 결함 감지
            defects = analyzer.analyze_image(img_info['filename'])
            
            # Chipping만 필터링
            chippings = [d for d in defects if d.defect_type == 'chipping']
            
            print(f"총 감지된 결함: {len(defects)}개")
            print(f"Chipping으로 분류된 결함: {len(chippings)}개")
            
            if chippings:
                print("\nChipping 상세 정보:")
                for i, chipping in enumerate(chippings, 1):
                    print(f"  Chipping {i}:")
                    print(f"    - 위치: ({chipping.centroid[0]}, {chipping.centroid[1]})")
                    print(f"    - 면적: {chipping.area:.2f} 픽셀")
                    print(f"    - 심각도: {chipping.severity}")
                    print(f"    - Solidity: {chipping.solidity:.3f}")
                    print(f"    - 종횡비: {chipping.aspect_ratio:.3f}")
            
            # 결과 시각화
            img_path = Path(img_info['filename'])
            result_filename = results_dir / f"{img_path.stem}_result.jpg"
            analyzer.visualize_results(
                img_info['filename'],
                defects,
                save_path=str(result_filename),
                show=False
            )
            print(f"결과 이미지 저장: {result_filename}\n")
            
        except Exception as e:
            print(f"오류 발생: {e}\n")
    
    print("=== 테스트 완료 ===")
    print(f"\n생성된 파일:")
    print(f"  - {generated_dir}/test_*.jpg: 원본 Chipping 이미지")
    print(f"  - {results_dir}/test_*_result.jpg: 감지 결과 이미지")


def create_sample_with_chipping():
    """샘플 Chipping 이미지 생성 (간단한 예제)"""
    print("\n=== 샘플 Chipping 이미지 생성 ===\n")
    
    # 폴더 설정
    generated_dir = Path('generated_images/chipping')
    results_dir = Path('test_results/chipping')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ChippingImageGenerator(width=1920, height=1080)
    
    # 다양한 위치에 Chipping 생성
    panel = generator.generate_base_panel()
    
    # 좌상 모서리 Chipping
    panel = generator.create_chipping(
        panel, (0, 0), (150, 150), 'corner', 0.4
    )
    
    # 상단 가장자리 Chipping
    panel = generator.create_chipping(
        panel, (500, 0), (300, 80), 'edge', 0.3
    )
    
    # 우하 모서리 Chipping
    panel = generator.create_chipping(
        panel, (1770, 930), (150, 150), 'corner', 0.5
    )
    
    sample_path = generated_dir / 'sample_chipping.jpg'
    generator.save_image(panel, str(sample_path))
    print(f"샘플 이미지 생성 완료: {sample_path}")
    
    # 감지 테스트
    print("\n감지 테스트 실행...")
    analyzer = DefectAnalyzer(min_defect_area=50)
    defects = analyzer.analyze_image(str(sample_path))
    
    chippings = [d for d in defects if d.defect_type == 'chipping']
    print(f"\n감지된 Chipping: {len(chippings)}개")
    
    result_path = results_dir / 'sample_chipping_result.jpg'
    analyzer.visualize_results(
        str(sample_path),
        defects,
        save_path=str(result_path),
        show=False
    )
    print(f"결과 이미지 저장: {result_path}")


if __name__ == '__main__':
    # 전체 테스트 실행
    test_chipping_detection()
    
    # 간단한 샘플 생성
    create_sample_with_chipping()

