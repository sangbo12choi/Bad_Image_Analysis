"""
불량별 이미지 분석 콘솔 프로그램
인터랙티브 메뉴를 통해 불량 유형별로 이미지를 분석할 수 있습니다.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from defect_analyzer import DefectAnalyzer, Defect


class DefectAnalysisConsole:
    """불량 분석 콘솔 프로그램"""
    
    def __init__(self):
        self.analyzer: Optional[DefectAnalyzer] = None
        self.defect_types = {
            '1': ('전체 불량', 'all'),
            '2': ('Chipping (가장자리/모서리 결함)', 'chipping'),
            '3': ('Crack (외곽에서 시작하는 균열)', 'crack'),
            '4': ('Scratch (선형 긁힘)', 'scratch'),
            '5': ('Bubble (기포)', 'bubble'),
            '6': ('사용자 정의 필터', 'custom')
        }
    
    def print_header(self):
        """헤더 출력"""
        print("=" * 70)
        print(" " * 15 + "불량별 이미지 분석 시스템")
        print("=" * 70)
        print()
    
    def print_menu(self):
        """메인 메뉴 출력"""
        print("\n[메인 메뉴]")
        print("-" * 70)
        print("1. 단일 이미지 분석")
        print("2. 배치 이미지 분석 (폴더)")
        print("3. 분석 설정 변경")
        print("4. 종료")
        print("-" * 70)
    
    def print_defect_type_menu(self):
        """불량 유형 선택 메뉴 출력"""
        print("\n[불량 유형 선택]")
        print("-" * 70)
        for key, (name, _) in self.defect_types.items():
            print(f"{key}. {name}")
        print("-" * 70)
    
    def get_user_input(self, prompt: str, valid_options: Optional[List[str]] = None) -> str:
        """사용자 입력 받기"""
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                if valid_options is None or user_input in valid_options:
                    return user_input
                else:
                    print(f"잘못된 입력입니다. 가능한 옵션: {', '.join(valid_options)}")
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                sys.exit(0)
            except EOFError:
                print("\n\n프로그램을 종료합니다.")
                sys.exit(0)
    
    def initialize_analyzer(self):
        """분석기 초기화"""
        if self.analyzer is None:
            print("\n분석기 초기화 중...")
            self.analyzer = DefectAnalyzer()
            print("✓ 분석기 초기화 완료")
        else:
            print("\n현재 분석기 설정:")
            print(f"  - 최소 결함 크기: {self.analyzer.min_defect_area} 픽셀")
            print(f"  - 최대 결함 크기: {self.analyzer.max_defect_area} 픽셀")
            print(f"  - 이진화 방법: {self.analyzer.threshold_method}")
    
    def change_settings(self):
        """분석 설정 변경"""
        print("\n[분석 설정 변경]")
        print("-" * 70)
        
        try:
            min_area = self.get_user_input(
                f"최소 결함 크기 (현재: {self.analyzer.min_defect_area if self.analyzer else 10})",
                None
            )
            min_area = int(min_area) if min_area else (self.analyzer.min_defect_area if self.analyzer else 10)
            
            max_area = self.get_user_input(
                f"최대 결함 크기 (현재: {self.analyzer.max_defect_area if self.analyzer else 100000})",
                None
            )
            max_area = int(max_area) if max_area else (self.analyzer.max_defect_area if self.analyzer else 100000)
            
            print("\n이진화 방법:")
            print("  1. adaptive (적응형)")
            print("  2. otsu (오츠)")
            print("  3. manual (수동)")
            threshold_choice = self.get_user_input("선택 (1-3)", ['1', '2', '3'])
            threshold_map = {'1': 'adaptive', '2': 'otsu', '3': 'manual'}
            threshold = threshold_map[threshold_choice]
            
            self.analyzer = DefectAnalyzer(
                min_defect_area=min_area,
                max_defect_area=max_area,
                threshold_method=threshold
            )
            
            print("\n✓ 설정이 변경되었습니다.")
            
        except ValueError:
            print("\n✗ 잘못된 입력입니다. 설정 변경이 취소되었습니다.")
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
    
    def filter_defects_by_type(self, defects: List[Defect], defect_type: str) -> List[Defect]:
        """불량 유형별로 필터링"""
        if defect_type == 'all':
            return defects
        return [d for d in defects if d.defect_type == defect_type]
    
    def analyze_single_image(self):
        """단일 이미지 분석"""
        print("\n[단일 이미지 분석]")
        print("-" * 70)
        
        image_path = self.get_user_input("이미지 파일 경로를 입력하세요")
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"✗ 파일을 찾을 수 없습니다: {image_path}")
            return
        
        if not image_path.is_file():
            print(f"✗ 파일이 아닙니다: {image_path}")
            return
        
        # 불량 유형 선택
        self.print_defect_type_menu()
        defect_choice = self.get_user_input("불량 유형을 선택하세요 (1-6)", ['1', '2', '3', '4', '5', '6'])
        defect_type_key = self.defect_types[defect_choice][1]
        
        # 출력 폴더 선택
        output_dir = self.get_user_input("결과 저장 폴더 (기본: output)", None)
        output_path = Path(output_dir if output_dir else 'output')
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\n이미지 분석 중: {image_path.name}")
            print("잠시만 기다려주세요...")
            
            # 이미지 분석
            all_defects = self.analyzer.analyze_image(str(image_path))
            
            # 불량 유형별 필터링
            if defect_type_key == 'custom':
                print("\n사용자 정의 필터:")
                print("  사용 가능한 불량 유형: chipping, crack, scratch, bubble, point, line, area, edge")
                custom_type = self.get_user_input("불량 유형 입력", None)
                filtered_defects = [d for d in all_defects if d.defect_type == custom_type.lower()]
            else:
                filtered_defects = self.filter_defects_by_type(all_defects, defect_type_key)
            
            # 결과 출력
            print(f"\n{'='*70}")
            print(f"분석 결과: {image_path.name}")
            print(f"{'='*70}")
            print(f"전체 감지된 불량: {len(all_defects)}개")
            print(f"선택한 유형의 불량: {len(filtered_defects)}개")
            
            if filtered_defects:
                print(f"\n[불량 상세 정보]")
                print("-" * 70)
                type_counts = {}
                for defect in filtered_defects:
                    defect_type = defect.defect_type
                    type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
                
                for defect_type, count in sorted(type_counts.items()):
                    print(f"  {defect_type}: {count}개")
                
                print(f"\n[심각도 분포]")
                severity_counts = {}
                for defect in filtered_defects:
                    severity = defect.severity
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                for severity, count in sorted(severity_counts.items()):
                    print(f"  {severity}: {count}개")
                
                # 결과 이미지 저장
                result_img_path = output_path / f"{image_path.stem}_result.jpg"
                self.analyzer.visualize_results(
                    str(image_path),
                    filtered_defects,
                    save_path=str(result_img_path),
                    show=False
                )
                print(f"\n✓ 결과 이미지 저장: {result_img_path}")
                
                # 리포트 저장
                report_path = output_path / f"{image_path.stem}_report.csv"
                self.analyzer.generate_report(filtered_defects, str(report_path))
                print(f"✓ 리포트 저장: {report_path}")
            else:
                print("\n선택한 유형의 불량이 감지되지 않았습니다.")
            
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_batch(self):
        """배치 이미지 분석"""
        print("\n[배치 이미지 분석]")
        print("-" * 70)
        
        input_dir = self.get_user_input("입력 폴더 경로를 입력하세요")
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"✗ 폴더를 찾을 수 없습니다: {input_path}")
            return
        
        if not input_path.is_dir():
            print(f"✗ 폴더가 아닙니다: {input_path}")
            return
        
        # 불량 유형 선택
        self.print_defect_type_menu()
        defect_choice = self.get_user_input("불량 유형을 선택하세요 (1-6)", ['1', '2', '3', '4', '5', '6'])
        defect_type_key = self.defect_types[defect_choice][1]
        
        # 출력 폴더 선택
        output_dir = self.get_user_input("결과 저장 폴더 (기본: output)", None)
        output_path = Path(output_dir if output_dir else 'output')
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"\n배치 처리 시작: {input_path}")
            print("잠시만 기다려주세요...")
            
            # 지원하는 이미지 확장자
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f'*{ext}'))
                image_files.extend(input_path.glob(f'*{ext.upper()}'))
            
            if not image_files:
                print(f"✗ 이미지 파일을 찾을 수 없습니다: {input_path}")
                return
            
            print(f"총 {len(image_files)}개의 이미지 발견")
            
            total_defects = 0
            processed_count = 0
            
            for img_file in image_files:
                try:
                    print(f"\n처리 중: {img_file.name}")
                    
                    # 이미지 분석
                    all_defects = self.analyzer.analyze_image(str(img_file))
                    
                    # 불량 유형별 필터링
                    if defect_type_key == 'custom':
                        print("  사용 가능한 불량 유형: chipping, crack, scratch, bubble, point, line, area, edge")
                        custom_type = self.get_user_input(f"  {img_file.name}의 불량 유형 입력", None)
                        filtered_defects = [d for d in all_defects if d.defect_type == custom_type.lower()]
                    else:
                        filtered_defects = self.filter_defects_by_type(all_defects, defect_type_key)
                    
                    if filtered_defects:
                        # 결과 이미지 저장
                        result_img_path = output_path / f"{img_file.stem}_result.jpg"
                        self.analyzer.visualize_results(
                            str(img_file),
                            filtered_defects,
                            save_path=str(result_img_path),
                            show=False
                        )
                        
                        # 리포트 저장
                        report_path = output_path / f"{img_file.stem}_report.csv"
                        self.analyzer.generate_report(filtered_defects, str(report_path))
                        
                        print(f"  ✓ 감지된 불량: {len(filtered_defects)}개")
                        total_defects += len(filtered_defects)
                    else:
                        print(f"  - 감지된 불량 없음")
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"  ✗ 오류 발생: {e}")
            
            print(f"\n{'='*70}")
            print(f"배치 처리 완료!")
            print(f"{'='*70}")
            print(f"처리된 이미지: {processed_count}/{len(image_files)}개")
            print(f"총 감지된 불량: {total_defects}개")
            print(f"결과 저장 위치: {output_path}")
            
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """콘솔 프로그램 실행"""
        self.print_header()
        self.initialize_analyzer()
        
        while True:
            self.print_menu()
            choice = self.get_user_input("메뉴를 선택하세요 (1-4)", ['1', '2', '3', '4'])
            
            if choice == '1':
                self.analyze_single_image()
            elif choice == '2':
                self.analyze_batch()
            elif choice == '3':
                self.change_settings()
            elif choice == '4':
                print("\n프로그램을 종료합니다.")
                break
            
            # 계속 진행할지 확인
            if choice != '4':
                continue_choice = self.get_user_input("\n계속하시겠습니까? (y/n)", ['y', 'n', 'Y', 'N'])
                if continue_choice.lower() == 'n':
                    print("\n프로그램을 종료합니다.")
                    break


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='불량별 이미지 분석 콘솔 프로그램',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 인터랙티브 모드 실행
  python analyze_console.py
  
  # 명령줄에서 직접 실행 (기존 main.py와 동일)
  python analyze_console.py image.jpg -o output
  
  # 배치 처리
  python analyze_console.py input_folder/ -o output_folder/
        """
    )
    
    parser.add_argument(
        'input',
        nargs='?',
        type=str,
        help='입력 이미지 파일 또는 폴더 경로 (없으면 인터랙티브 모드)'
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
        '--type',
        type=str,
        choices=['all', 'chipping', 'crack', 'scratch', 'bubble'],
        default='all',
        help='분석할 불량 유형 (기본값: all)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='결과 이미지를 화면에 표시하지 않음'
    )
    
    args = parser.parse_args()
    
    # 명령줄 인자가 있으면 기존 main.py 방식으로 실행
    if args.input:
        from defect_analyzer import DefectAnalyzer
        from pathlib import Path
        
        analyzer = DefectAnalyzer(
            min_defect_area=args.min_area,
            max_defect_area=args.max_area,
            threshold_method=args.threshold
        )
        
        input_path = Path(args.input)
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if input_path.is_file():
            print(f"이미지 분석 중: {input_path}")
            all_defects = analyzer.analyze_image(str(input_path))
            
            # 불량 유형 필터링
            if args.type != 'all':
                filtered_defects = [d for d in all_defects if d.defect_type == args.type]
            else:
                filtered_defects = all_defects
            
            # 결과 시각화
            result_img_path = output_path / f"{input_path.stem}_result.jpg"
            analyzer.visualize_results(
                str(input_path),
                filtered_defects,
                save_path=str(result_img_path),
                show=not args.no_show
            )
            
            # 리포트 생성
            report_path = output_path / f"{input_path.stem}_report.csv"
            analyzer.generate_report(filtered_defects, str(report_path))
            
            print(f"\n총 {len(filtered_defects)}개의 결함이 감지되었습니다.")
            print(f"결과 저장 위치: {output_path}")
            
        elif input_path.is_dir():
            print(f"배치 처리 시작: {input_path}")
            analyzer.batch_analyze(str(input_path), str(output_path))
            print("\n배치 처리 완료!")
        else:
            print(f"오류: 입력 경로를 찾을 수 없습니다: {input_path}")
    else:
        # 인터랙티브 모드 실행
        console = DefectAnalysisConsole()
        console.run()


if __name__ == '__main__':
    main()

