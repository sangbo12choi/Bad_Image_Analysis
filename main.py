"""
Panel Hard Defect Analysis - Main Script
사용 예제 및 메인 실행 스크립트
"""

import argparse
from pathlib import Path
from defect_analyzer import DefectAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='디스플레이 패널 Hard Defect 분석 도구'
    )
    parser.add_argument(
        'input',
        type=str,
        help='입력 이미지 파일 또는 폴더 경로'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='결과 저장 폴더 (기본값: output)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=10,
        help='최소 결함 크기 (픽셀, 기본값: 10)'
    )
    parser.add_argument(
        '--max-area',
        type=int,
        default=100000,
        help='최대 결함 크기 (픽셀, 기본값: 100000)'
    )
    parser.add_argument(
        '--threshold',
        type=str,
        choices=['adaptive', 'otsu', 'manual'],
        default='adaptive',
        help='이진화 방법 (기본값: adaptive)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='결과 이미지를 화면에 표시하지 않음'
    )
    
    args = parser.parse_args()
    
    # 분석기 생성
    analyzer = DefectAnalyzer(
        min_defect_area=args.min_area,
        max_defect_area=args.max_area,
        threshold_method=args.threshold
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 단일 이미지 또는 배치 처리
    if input_path.is_file():
        print(f"이미지 분석 중: {input_path}")
        defects = analyzer.analyze_image(str(input_path))
        
        # 결과 시각화
        result_img_path = output_path / f"{input_path.stem}_result.jpg"
        analyzer.visualize_results(
            str(input_path), 
            defects,
            save_path=str(result_img_path),
            show=not args.no_show
        )
        
        # 리포트 생성
        report_path = output_path / f"{input_path.stem}_report.csv"
        analyzer.generate_report(defects, str(report_path))
        
        print(f"\n총 {len(defects)}개의 결함이 감지되었습니다.")
        
    elif input_path.is_dir():
        print(f"배치 처리 시작: {input_path}")
        analyzer.batch_analyze(str(input_path), str(output_path))
        print("\n배치 처리 완료!")
    else:
        print(f"오류: 입력 경로를 찾을 수 없습니다: {input_path}")


if __name__ == '__main__':
    main()

