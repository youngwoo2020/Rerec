import os
import shutil

# ===== 변수 =====
recent_month = ''
k_core = 3
k_reps = 5
dataset = 'beauty'

src_dir = "data/handled"
dst_dir = f"data/{dataset}/handled"

# 매핑: {표준파일명: 원래 suffix}
file_map = {
    "title_seq.txt": "title_seq.txt",
    "review_seq.txt": "review_seq.txt",
    # "rating_seq.txt": "rating_seq.txt",
    "inter_seq.txt": "inter_seq.txt",
    "id_map.json": "id_map.json",
    "items2rep_sentences.json": "items2rep_sentences.json",
    "item2attributes.json": "item2attributes.json",
    "inter.txt": "inter.txt",
    "user_items.json": "user_items.json",
    # "rating.txt": "rating.txt",
    # "title.txt": "title.txt",
    "review.txt": "review.txt",
    "user_items.json": "user_items.json"
}

# ===== 실행 =====
for new_name, suffix in file_map.items():
    old_name = f"{recent_month}kcore{k_core}_kreps{k_reps}_{suffix}"
    src_path = os.path.join(src_dir, old_name)
    dst_path = os.path.join(dst_dir, new_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"✓ {old_name} → {new_name}")
    else:
        print(f"✗ 파일 없음: {old_name}")

print("\n=== 이동 완료 ===")
