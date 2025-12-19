"""
Scratch 불량 이미지 생성 및 감지 - 간단한 실행 예제
이 스크립트를 실행하면 Scratch 이미지를 생성하고 감지합니다.
"""

from scratch_generator import ScratchImageGenerator
from defect_analyzer import DefectAnalyzer
from pathlib import Path
import math


def main():
    """Scratch 이미지 생성 및 감지 메인 함수"""
    
    print("=" * 60)
    print("Scratch 불량 이미지 생성 및 감지 프로그램")
    print("=" * 60)
    print()
    
    # 폴더 설정
    generated_dir = Path('generated_images/scratch')
    results_dir = Path('test_results/scratch')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1단계: Scratch 이미지 생성
    # ==========================================
    print("1단계: Scratch 이미지 생성 중...")
    print("-" * 60)
    
    generator = ScratchImageGenerator(width=1920, height=1080)
    
    # 기본 패널 생성
    panel = generator.generate_base_panel()
    
    # 다양한 위치에 Scratch 생성 (패널 표면 어디서나)
    print("  - 중앙에서 시작하는 직선 Scratch 생성...")
    panel = generator.create_scratch(
        panel, 
        start_point=(500, 400),      # 중앙 근처
        length=300,                  # 길이
        direction=math.pi / 4,        # 대각선 방향
        scratch_type='straight',     # 직선 형태
        width=1,                     # 두께 (Scratch는 얇음)
        depth=0.5                    # 깊이/어둠 정도
    )
    
    print("  - 좌측 상단에서 시작하는 곡선 Scratch 생성...")
    panel = generator.create_scratch(
        panel, 
        start_point=(200, 200),      # 좌측 상단
        length=250, 
        direction=0.3,                # 오른쪽 아래 방향
        scratch_type='curved',       # 곡선 형태
        width=1, 
        depth=0.6
    )
    
    print("  - 우측 하단에서 시작하는 지그재그 Scratch 생성...")
    panel = generator.create_scratch(
        panel, 
        start_point=(1500, 800),     # 우측 하단
        length=350, 
        direction=-math.pi / 3,      # 왼쪽 위 방향
        scratch_type='zigzag',       # 지그재그 형태
        width=2, 
        depth=0.4
    )
    
    # 이미지 저장
    image_path = generated_dir / 'scratch_panel.jpg'
    generator.save_image(panel, str(image_path))
    print(f"\n✓ 이미지 저장 완료: {image_path}")
    print()
    
    # ==========================================
    # 2단계: Scratch 감지
    # ==========================================
    print("2단계: Scratch 감지 중...")
    print("-" * 60)
    
    analyzer = DefectAnalyzer(
        min_defect_area=20,      # 최소 결함 크기 (Scratch는 얇음)
        max_defect_area=10000,   # 최대 결함 크기
        threshold_method='adaptive'  # 이진화 방법
    )
    
    defects = analyzer.analyze_image(str(image_path))
    
    # Scratch만 필터링
    scratches = [d for d in defects if d.defect_type == 'scratch']
    
    print(f"  - 총 감지된 결함: {len(defects)}개")
    print(f"  - Scratch로 분류된 결함: {len(scratches)}개")
    print()
    
    # Scratch 상세 정보 출력
    if scratches:
        print("  Scratch 상세 정보:")
        for i, scratch in enumerate(scratches, 1):
            print(f"\n    Scratch {i}:")
            print(f"      위치: ({scratch.centroid[0]:.0f}, {scratch.centroid[1]:.0f})")
            print(f"      면적: {scratch.area:.2f} 픽셀")
            print(f"      둘레: {scratch.perimeter:.2f} 픽셀")
            print(f"      종횡비: {scratch.aspect_ratio:.3f}")
            print(f"      심각도: {scratch.severity}")
            print(f"      Solidity: {scratch.solidity:.3f}")
    else:
        print("  ⚠ Scratch가 감지되지 않았습니다.")
        print("    - min_defect_area 값을 낮춰보세요 (예: 15)")
        print("    - threshold_method를 'otsu'로 변경해보세요")
    print()
    
    # ==========================================
    # 3단계: 결과 시각화
    # ==========================================
    print("3단계: 결과 시각화 중...")
    print("-" * 60)
    
    result_path = results_dir / 'scratch_result.jpg'
    analyzer.visualize_results(
        str(image_path),
        defects,
        save_path=str(result_path),
        show=False  # True로 변경하면 화면에 표시됨
    )
    print(f"✓ 결과 이미지 저장: {result_path}")
    print()
    
    # ==========================================
    # 4단계: 리포트 생성
    # ==========================================
    print("4단계: 리포트 생성 중...")
    print("-" * 60)
    
    report_path = results_dir / 'scratch_report.csv'
    analyzer.generate_report(defects, str(report_path))
    print(f"✓ 리포트 저장: {report_path}")
    print()
    
    # ==========================================
    # 완료
    # ==========================================
    print("=" * 60)
    print("✓ 모든 작업 완료!")
    print("=" * 60)
    print()
    print("생성된 파일:")
    print(f"  - 원본 이미지: {image_path}")
    print(f"  - 결과 이미지: {result_path}")
    print(f"  - 리포트: {report_path}")
    print()
    print("다음 단계:")
    print("  - 결과 이미지를 확인하여 Scratch가 주황색(orange)으로 표시되었는지 확인하세요")
    print("  - 리포트 CSV 파일에서 상세 정보를 확인할 수 있습니다")
    print("  - 파라미터를 조정하려면 이 스크립트를 수정하세요")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\n문제 해결:")
        print("  1. 필수 패키지가 설치되어 있는지 확인: pip install -r requirements.txt")
        print("  2. 폴더 구조가 올바른지 확인: python setup_folders.py")
        import traceback
        traceback.print_exc()

