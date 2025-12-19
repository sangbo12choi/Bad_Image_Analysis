"""
Scratch 감지 테스트 스크립트
가상의 Scratch 이미지를 생성하고 감지 기능을 테스트합니다.
Scratch는 패널 표면 어디서나 발생할 수 있습니다.
"""

from scratch_generator import ScratchImageGenerator
from defect_analyzer import DefectAnalyzer
from pathlib import Path


def test_scratch_detection():
    """Scratch 감지 테스트"""
    print("=== Scratch 감지 테스트 ===\n")
    
    # 폴더 설정
    generated_dir = Path('generated_images/scratch')
    results_dir = Path('test_results/scratch')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scratch 이미지 생성
    print("1. 가상 Scratch 이미지 생성 중...")
    generator = ScratchImageGenerator(width=1920, height=1080)
    
    # 다양한 유형의 Scratch 생성
    test_cases = [
        {
            'name': 'straight_scratch',
            'num_scratches': 1,
            'types': ['straight'],
            'description': '직선 Scratch'
        },
        {
            'name': 'curved_scratch',
            'num_scratches': 1,
            'types': ['curved'],
            'description': '곡선 Scratch'
        },
        {
            'name': 'zigzag_scratch',
            'num_scratches': 1,
            'types': ['zigzag'],
            'description': '지그재그 Scratch'
        },
        {
            'name': 'multiple_scratches',
            'num_scratches': 3,
            'types': ['straight', 'curved', 'zigzag'],
            'description': '다중 Scratch'
        }
    ]
    
    generated_images = []
    
    for test_case in test_cases:
        print(f"   - {test_case['description']} 생성 중...")
        panel = generator.generate_panel_with_scratch(
            num_scratches=test_case['num_scratches'],
            scratch_types=test_case['types'],
            min_length=150,
            max_length=400
        )
        
        filename = generated_dir / f"test_{test_case['name']}.jpg"
        generator.save_image(panel, str(filename))
        generated_images.append({
            'filename': str(filename),
            'description': test_case['description']
        })
    
    print("\n2. Scratch 감지 테스트 시작...\n")
    
    # 2. Scratch 감지
    analyzer = DefectAnalyzer(
        min_defect_area=20,  # Scratch는 얇지만 길 수 있음
        max_defect_area=10000,
        threshold_method='adaptive'
    )
    
    for img_info in generated_images:
        print(f"--- {img_info['description']} 분석 ---")
        print(f"이미지: {img_info['filename']}")
        
        try:
            # 결함 감지
            defects = analyzer.analyze_image(img_info['filename'])
            
            # Scratch만 필터링
            scratches = [d for d in defects if d.defect_type == 'scratch']
            
            print(f"총 감지된 결함: {len(defects)}개")
            print(f"Scratch로 분류된 결함: {len(scratches)}개")
            
            if scratches:
                print("\nScratch 상세 정보:")
                for i, scratch in enumerate(scratches, 1):
                    print(f"  Scratch {i}:")
                    print(f"    - 위치: ({scratch.centroid[0]}, {scratch.centroid[1]})")
                    print(f"    - 면적: {scratch.area:.2f} 픽셀")
                    print(f"    - 둘레: {scratch.perimeter:.2f} 픽셀")
                    print(f"    - 종횡비: {scratch.aspect_ratio:.3f}")
                    print(f"    - 심각도: {scratch.severity}")
                    print(f"    - Solidity: {scratch.solidity:.3f}")
            
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
    print(f"  - {generated_dir}/test_*.jpg: 원본 Scratch 이미지")
    print(f"  - {results_dir}/test_*_result.jpg: 감지 결과 이미지")


def create_sample_with_scratch():
    """샘플 Scratch 이미지 생성 (간단한 예제)"""
    print("\n=== 샘플 Scratch 이미지 생성 ===\n")
    
    # 폴더 설정
    generated_dir = Path('generated_images/scratch')
    results_dir = Path('test_results/scratch')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    generator = ScratchImageGenerator(width=1920, height=1080)
    
    # 다양한 위치에 Scratch 생성 (패널 표면 어디서나)
    panel = generator.generate_base_panel()
    
    # 중앙에서 시작하는 직선 Scratch
    import math
    panel = generator.create_scratch(
        panel, (500, 400), 300, math.pi / 4, 'straight', 1, 0.5
    )
    
    # 좌측 상단에서 시작하는 곡선 Scratch
    panel = generator.create_scratch(
        panel, (200, 200), 250, 0.3, 'curved', 1, 0.6
    )
    
    # 우측 하단에서 시작하는 지그재그 Scratch
    panel = generator.create_scratch(
        panel, (1500, 800), 350, -math.pi / 3, 'zigzag', 2, 0.4
    )
    
    sample_path = generated_dir / 'sample_scratch.jpg'
    generator.save_image(panel, str(sample_path))
    print(f"샘플 이미지 생성 완료: {sample_path}")
    
    # 감지 테스트
    print("\n감지 테스트 실행...")
    analyzer = DefectAnalyzer(min_defect_area=20)
    defects = analyzer.analyze_image(str(sample_path))
    
    scratches = [d for d in defects if d.defect_type == 'scratch']
    print(f"\n감지된 Scratch: {len(scratches)}개")
    
    # 모든 결함 유형 출력
    print("\n감지된 모든 결함 유형:")
    defect_types = {}
    for defect in defects:
        defect_types[defect.defect_type] = defect_types.get(defect.defect_type, 0) + 1
    for defect_type, count in defect_types.items():
        print(f"  - {defect_type}: {count}개")
    
    result_path = results_dir / 'sample_scratch_result.jpg'
    analyzer.visualize_results(
        str(sample_path),
        defects,
        save_path=str(result_path),
        show=False
    )
    print(f"결과 이미지 저장: {result_path}")


if __name__ == '__main__':
    # 전체 테스트 실행
    test_scratch_detection()
    
    # 간단한 샘플 생성
    create_sample_with_scratch()

