"""
폴더 구조 설정 스크립트
프로젝트에 필요한 폴더 구조를 생성합니다.
"""

from pathlib import Path


def setup_folders():
    """필요한 폴더 구조 생성"""
    folders = [
        'generated_images/chipping',  # Chipping 가상 생성 이미지
        'generated_images/crack',     # Crack 가상 생성 이미지
        'generated_images/scratch',   # Scratch 가상 생성 이미지
        'test_results/chipping',      # Chipping 테스트 결과
        'test_results/crack',         # Crack 테스트 결과
        'test_results/scratch',       # Scratch 테스트 결과
        'output',                     # 일반 분석 결과 (이미 있을 수 있음)
    ]
    
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"폴더 생성/확인: {folder}")
    
    # .gitkeep 파일 생성 (빈 폴더도 Git에 포함되도록)
    for folder in folders:
        gitkeep = Path(folder) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
    
    print("\n폴더 구조 설정 완료!")


if __name__ == '__main__':
    setup_folders()

