"""
Crack 감지 테스트 스크립트
가상의 Crack 이미지를 생성하고 감지 기능을 테스트합니다.
Crack은 Panel의 외곽에서만 발생합니다.
"""

from crack_generator import CrackImageGenerator
from defect_analyzer import DefectAnalyzer
from pathlib import Path


def test_crack_detection():
    """Crack 감지 테스트"""
    print("=== Crack 감지 테스트 ===\n")
    
    # 폴더 설정
    generated_dir = Path('generated_images/crack')
    results_dir = Path('test_results/crack')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Crack 이미지 생성
    print("1. 가상 Crack 이미지 생성 중...")
    generator = CrackImageGenerator(width=1920, height=1080)
    
    # 다양한 유형의 Crack 생성
    test_cases = [
        {
            'name': 'straight_crack',
            'num_cracks': 1,
            'types': ['straight'],
            'description': '직선 Crack'
        },
        {
            'name': 'curved_crack',
            'num_cracks': 1,
            'types': ['curved'],
            'description': '곡선 Crack'
        },
        {
            'name': 'branching_crack',
            'num_cracks': 1,
            'types': ['branching'],
            'description': '분기 Crack'
        },
        {
            'name': 'multiple_cracks',
            'num_cracks': 3,
            'types': ['straight', 'curved', 'branching'],
            'description': '다중 Crack'
        }
    ]
    
    generated_images = []
    
    for test_case in test_cases:
        print(f"   - {test_case['description']} 생성 중...")
        panel = generator.generate_panel_with_crack(
            num_cracks=test_case['num_cracks'],
            crack_types=test_case['types'],
            min_length=150,
            max_length=400
        )
        
        filename = generated_dir / f"test_{test_case['name']}.jpg"
        generator.save_image(panel, str(filename))
        generated_images.append({
            'filename': str(filename),
            'description': test_case['description']
        })
    
    print("\n2. Crack 감지 테스트 시작...\n")
    
    # 2. Crack 감지
    analyzer = DefectAnalyzer(
        min_defect_area=30,  # Crack은 얇지만 길 수 있음
        max_defect_area=20000,
        threshold_method='adaptive'
    )
    
    for img_info in generated_images:
        print(f"--- {img_info['description']} 분석 ---")
        print(f"이미지: {img_info['filename']}")
        
        try:
            # 결함 감지
            defects = analyzer.analyze_image(img_info['filename'])
            
            # Crack만 필터링
            cracks = [d for d in defects if d.defect_type == 'crack']
            
            print(f"총 감지된 결함: {len(defects)}개")
            print(f"Crack으로 분류된 결함: {len(cracks)}개")
            
            if cracks:
                print("\nCrack 상세 정보:")
                for i, crack in enumerate(cracks, 1):
                    print(f"  Crack {i}:")
                    print(f"    - 위치: ({crack.centroid[0]}, {crack.centroid[1]})")
                    print(f"    - 면적: {crack.area:.2f} 픽셀")
                    print(f"    - 둘레: {crack.perimeter:.2f} 픽셀")
                    print(f"    - 종횡비: {crack.aspect_ratio:.3f}")
                    print(f"    - 심각도: {crack.severity}")
                    print(f"    - Solidity: {crack.solidity:.3f}")
            
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
    print(f"  - {generated_dir}/test_*.jpg: 원본 Crack 이미지")
    print(f"  - {results_dir}/test_*_result.jpg: 감지 결과 이미지")


def create_sample_with_crack():
    """샘플 Crack 이미지 생성 (간단한 예제)"""
    print("\n=== 샘플 Crack 이미지 생성 ===\n")
    
    # 폴더 설정
    generated_dir = Path('generated_images/crack')
    results_dir = Path('test_results/crack')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    generator = CrackImageGenerator(width=1920, height=1080)
    
    # 다양한 위치에 Crack 생성 (모두 외곽에서 시작)
    panel = generator.generate_base_panel()
    
    # 상단 가장자리에서 시작하는 직선 Crack
    import math
    panel = generator.create_crack(
        panel, (200, 10), 300, math.pi / 2, 'straight', 2, 0.5
    )
    
    # 좌측 가장자리에서 시작하는 곡선 Crack
    panel = generator.create_crack(
        panel, (10, 300), 250, 0.2, 'curved', 2, 0.6
    )
    
    # 하단 가장자리에서 시작하는 분기 Crack
    panel = generator.create_crack(
        panel, (1500, 1070), 350, -math.pi / 2, 'branching', 3, 0.5
    )
    
    sample_path = generated_dir / 'sample_crack.jpg'
    generator.save_image(panel, str(sample_path))
    print(f"샘플 이미지 생성 완료: {sample_path}")
    
    # 감지 테스트
    print("\n감지 테스트 실행...")
    analyzer = DefectAnalyzer(min_defect_area=30)
    defects = analyzer.analyze_image(str(sample_path))
    
    cracks = [d for d in defects if d.defect_type == 'crack']
    print(f"\n감지된 Crack: {len(cracks)}개")
    
    # 모든 결함 유형 출력
    print("\n감지된 모든 결함 유형:")
    defect_types = {}
    for defect in defects:
        defect_types[defect.defect_type] = defect_types.get(defect.defect_type, 0) + 1
    for defect_type, count in defect_types.items():
        print(f"  - {defect_type}: {count}개")
    
    result_path = results_dir / 'sample_crack_result.jpg'
    analyzer.visualize_results(
        str(sample_path),
        defects,
        save_path=str(result_path),
        show=False
    )
    print(f"결과 이미지 저장: {result_path}")


if __name__ == '__main__':
    # 전체 테스트 실행
    test_crack_detection()
    
    # 간단한 샘플 생성
    create_sample_with_crack()

