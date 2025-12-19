"""생성된 이미지 파일 경로 출력 스크립트"""

from pathlib import Path

base = Path('generated_images')
chipping_dir = base / 'chipping' / 'test'
crack_dir = base / 'crack' / 'test'
scratch_dir = base / 'scratch' / 'test'

print("=" * 70)
print("생성된 이미지 파일 경로")
print("=" * 70)

# 절대 경로
print(f"\n[기본 디렉토리]")
print(f"  {base.absolute()}")
print(f"\n[상대 경로]")
print(f"  {base}")

# Chipping 이미지
print(f"\n[Chipping 이미지] ({len(list(chipping_dir.glob('*.jpg')))}개)")
print(f"  디렉토리: {chipping_dir.absolute()}")
print(f"  상대 경로: {chipping_dir}")
print(f"\n  파일 목록:")
for f in sorted(chipping_dir.glob('*.jpg')):
    print(f"    - {f.name}")

# Crack 이미지
print(f"\n[Crack 이미지] ({len(list(crack_dir.glob('*.jpg')))}개)")
print(f"  디렉토리: {crack_dir.absolute()}")
print(f"  상대 경로: {crack_dir}")
print(f"\n  파일 목록:")
for f in sorted(crack_dir.glob('*.jpg')):
    print(f"    - {f.name}")

# Scratch 이미지
print(f"\n[Scratch 이미지] ({len(list(scratch_dir.glob('*.jpg')))}개)")
print(f"  디렉토리: {scratch_dir.absolute()}")
print(f"  상대 경로: {scratch_dir}")
print(f"\n  파일 목록:")
for f in sorted(scratch_dir.glob('*.jpg')):
    print(f"    - {f.name}")

print("\n" + "=" * 70)
print(f"총 생성된 이미지: {len(list(chipping_dir.glob('*.jpg'))) + len(list(crack_dir.glob('*.jpg'))) + len(list(scratch_dir.glob('*.jpg')))}개")
print("=" * 70)

