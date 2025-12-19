"""
기존 이미지 파일들을 새 폴더 구조로 정리하는 스크립트
"""

from pathlib import Path
import shutil


def organize_existing_images():
    """기존 루트 디렉토리의 이미지 파일들을 새 폴더 구조로 이동"""
    
    # 폴더 생성
    generated_chipping = Path('generated_images/chipping')
    generated_crack = Path('generated_images/crack')
    results_chipping = Path('test_results/chipping')
    results_crack = Path('test_results/crack')
    
    for folder in [generated_chipping, generated_crack, results_chipping, results_crack]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # 현재 디렉토리의 이미지 파일 찾기
    root = Path('.')
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    moved_count = 0
    
    for img_file in root.glob('*'):
        if img_file.suffix.lower() in image_extensions:
            filename = img_file.name.lower()
            
            # Chipping 관련 파일
            if 'chipping' in filename and 'result' not in filename:
                dest = generated_chipping / img_file.name
                shutil.move(str(img_file), str(dest))
                print(f"이동: {img_file.name} -> {dest}")
                moved_count += 1
            elif 'chipping' in filename and 'result' in filename:
                dest = results_chipping / img_file.name
                shutil.move(str(img_file), str(dest))
                print(f"이동: {img_file.name} -> {dest}")
                moved_count += 1
            
            # Crack 관련 파일
            elif 'crack' in filename and 'result' not in filename:
                dest = generated_crack / img_file.name
                shutil.move(str(img_file), str(dest))
                print(f"이동: {img_file.name} -> {dest}")
                moved_count += 1
            elif 'crack' in filename and 'result' in filename:
                dest = results_crack / img_file.name
                shutil.move(str(img_file), str(dest))
                print(f"이동: {img_file.name} -> {dest}")
                moved_count += 1
            
            # test_로 시작하는 파일
            elif filename.startswith('test_'):
                if 'chipping' in filename:
                    if 'result' in filename:
                        dest = results_chipping / img_file.name
                    else:
                        dest = generated_chipping / img_file.name
                elif 'crack' in filename:
                    if 'result' in filename:
                        dest = results_crack / img_file.name
                    else:
                        dest = generated_crack / img_file.name
                else:
                    # 일반 테스트 이미지는 generated_images로
                    dest = generated_chipping / img_file.name
                
                shutil.move(str(img_file), str(dest))
                print(f"이동: {img_file.name} -> {dest}")
                moved_count += 1
            
            # sample_로 시작하는 파일
            elif filename.startswith('sample_'):
                if 'chipping' in filename:
                    dest = generated_chipping / img_file.name
                elif 'crack' in filename:
                    dest = generated_crack / img_file.name
                else:
                    dest = generated_chipping / img_file.name
                
                shutil.move(str(img_file), str(dest))
                print(f"이동: {img_file.name} -> {dest}")
                moved_count += 1
    
    print(f"\n총 {moved_count}개의 파일을 이동했습니다.")
    print("\n폴더 구조:")
    print("  - generated_images/chipping/: Chipping 샘플 이미지")
    print("  - generated_images/crack/: Crack 샘플 이미지")
    print("  - test_results/chipping/: Chipping 테스트 결과")
    print("  - test_results/crack/: Crack 테스트 결과")


if __name__ == '__main__':
    print("기존 이미지 파일 정리 중...\n")
    organize_existing_images()
    print("\n정리 완료!")

