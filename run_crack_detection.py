"""
Crack 불량 이미지 생성 및 감지 - 간단한 실행 예제
이 스크립트를 실행하면 Crack 이미지를 생성하고 감지합니다.
"""

from crack_generator import CrackImageGenerator
from defect_analyzer import DefectAnalyzer
from pathlib import Path
import math


def main():
    """Crack 이미지 생성 및 감지 메인 함수"""
    
    print("=" * 60)
    print("Crack 불량 이미지 생성 및 감지 프로그램")
    print("=" * 60)
    print()
    
    # 폴더 설정
    generated_dir = Path('generated_images/crack')
    results_dir = Path('test_results/crack')
    generated_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1단계: Crack 이미지 생성
    # ==========================================
    print("1단계: Crack 이미지 생성 중...")
    print("-" * 60)
    
    generator = CrackImageGenerator(width=1920, height=1080)
    
    # 기본 패널 생성
    panel = generator.generate_base_panel()
    
    # 다양한 위치에 Crack 생성 (모두 외곽에서 시작)
    print("  - 상단 가장자리에서 시작하는 직선 Crack 생성...")
    panel = generator.create_crack(
        panel, 
        start_point=(200, 10),      # 상단 가장자리
        length=300,                 # 길이
        direction=math.pi / 2,      # 아래쪽 방향
        crack_type='straight',      # 직선 형태
        width=2,                    # 두께
        depth=0.5                  # 깊이
    )
    
    print("  - 좌측 가장자리에서 시작하는 곡선 Crack 생성...")
    panel = generator.create_crack(
        panel, 
        start_point=(10, 300),       # 좌측 가장자리
        length=250, 
        direction=0.2,               # 오른쪽 방향
        crack_type='curved',         # 곡선 형태
        width=2, 
        depth=0.6
    )
    
    print("  - 하단 가장자리에서 시작하는 분기 Crack 생성...")
    panel = generator.create_crack(
        panel, 
        start_point=(1500, 1070),    # 하단 가장자리
        length=350, 
        direction=-math.pi / 2,      # 위쪽 방향
        crack_type='branching',       # 분기 형태
        width=3, 
        depth=0.5
    )
    
    # 이미지 저장
    image_path = generated_dir / 'crack_panel.jpg'
    generator.save_image(panel, str(image_path))
    print(f"\n✓ 이미지 저장 완료: {image_path}")
    print()
    
    # ==========================================
    # 2단계: Crack 감지
    # ==========================================
    print("2단계: Crack 감지 중...")
    print("-" * 60)
    
    analyzer = DefectAnalyzer(
        min_defect_area=30,      # 최소 결함 크기
        max_defect_area=20000,   # 최대 결함 크기
        threshold_method='adaptive'  # 이진화 방법
    )
    
    defects = analyzer.analyze_image(str(image_path))
    
    # Crack만 필터링
    cracks = [d for d in defects if d.defect_type == 'crack']
    
    print(f"  - 총 감지된 결함: {len(defects)}개")
    print(f"  - Crack으로 분류된 결함: {len(cracks)}개")
    print()
    
    # Crack 상세 정보 출력
    if cracks:
        print("  Crack 상세 정보:")
        for i, crack in enumerate(cracks, 1):
            print(f"\n    Crack {i}:")
            print(f"      위치: ({crack.centroid[0]:.0f}, {crack.centroid[1]:.0f})")
            print(f"      면적: {crack.area:.2f} 픽셀")
            print(f"      둘레: {crack.perimeter:.2f} 픽셀")
            print(f"      종횡비: {crack.aspect_ratio:.3f}")
            print(f"      심각도: {crack.severity}")
            print(f"      Solidity: {crack.solidity:.3f}")
    else:
        print("  ⚠ Crack이 감지되지 않았습니다.")
        print("    - min_defect_area 값을 낮춰보세요 (예: 20)")
        print("    - threshold_method를 'otsu'로 변경해보세요")
    print()
    
    # ==========================================
    # 3단계: 결과 시각화
    # ==========================================
    print("3단계: 결과 시각화 중...")
    print("-" * 60)
    
    result_path = results_dir / 'crack_result.jpg'
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
    
    report_path = results_dir / 'crack_report.csv'
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
    print("  - 결과 이미지를 확인하여 Crack이 청록색(cyan)으로 표시되었는지 확인하세요")
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

