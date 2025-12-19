"""
Panel Hard Defect Analysis - 사용 예제
"""

from defect_analyzer import DefectAnalyzer


def example_single_image():
    """단일 이미지 분석 예제"""
    print("=== 단일 이미지 분석 예제 ===\n")
    
    # 분석기 생성 (파라미터 커스터마이징 가능)
    analyzer = DefectAnalyzer(
        min_defect_area=10,      # 최소 결함 크기
        max_defect_area=100000,  # 최대 결함 크기
        threshold_method='adaptive'  # 이진화 방법
    )
    
    # 이미지 분석
    from pathlib import Path
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    image_path = 'sample_panel.jpg'  # 실제 이미지 경로로 변경하세요
    try:
        defects = analyzer.analyze_image(image_path)
        
        # 결과 시각화
        analyzer.visualize_results(
            image_path, 
            defects,
            save_path=str(output_dir / 'result.jpg'),
            show=True
        )
        
        # 리포트 생성
        analyzer.generate_report(defects, str(output_dir / 'report.csv'))
        
        print(f"\n감지된 결함 수: {len(defects)}")
        for i, defect in enumerate(defects, 1):
            print(f"\n결함 {i}:")
            print(f"  - 유형: {defect.defect_type}")
            print(f"  - 심각도: {defect.severity}")
            print(f"  - 면적: {defect.area:.2f} 픽셀")
            print(f"  - 위치: ({defect.centroid[0]}, {defect.centroid[1]})")
            
    except FileNotFoundError:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        print("실제 이미지 경로로 변경하거나 샘플 이미지를 준비해주세요.")


def example_batch_processing():
    """배치 처리 예제"""
    print("\n=== 배치 처리 예제 ===\n")
    
    analyzer = DefectAnalyzer()
    
    # 폴더 내 모든 이미지 분석
    input_folder = 'input_images/'  # 입력 폴더 경로
    output_folder = 'output_results/'  # 출력 폴더 경로
    
    try:
        analyzer.batch_analyze(input_folder, output_folder)
        print("\n배치 처리 완료!")
    except Exception as e:
        print(f"오류 발생: {e}")


def example_custom_parameters():
    """커스텀 파라미터 사용 예제"""
    print("\n=== 커스텀 파라미터 예제 ===\n")
    
    # 작은 결함도 감지하도록 설정
    analyzer_sensitive = DefectAnalyzer(
        min_defect_area=5,  # 더 작은 결함도 감지
        threshold_method='otsu'  # Otsu 이진화 사용
    )
    
    # 큰 결함만 감지하도록 설정
    analyzer_large_only = DefectAnalyzer(
        min_defect_area=500,  # 큰 결함만
        max_defect_area=50000
    )
    
    print("민감한 설정으로 분석기 생성 완료")
    print("큰 결함만 감지하는 설정으로 분석기 생성 완료")


if __name__ == '__main__':
    # 예제 실행
    example_single_image()
    # example_batch_processing()
    # example_custom_parameters()

