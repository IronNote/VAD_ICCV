import numpy as np
import os
import re

# 📂 GT 데이터가 저장된 폴더
input_dir = "C:/Myoung_Chul_KIM/VadCLIP(AAAI_2024)/VadCLIP-main/list/category_GT/video_GT"
output_dir = "C:/Myoung_Chul_KIM/VadCLIP(AAAI_2024)/VadCLIP-main/list/category_GT"

# ✅ 저장 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# ✅ 카테고리별 GT 데이터를 저장할 딕셔너리
category_gt = {}

# 📂 GT 폴더 내 모든 파일을 순회
for file_name in os.listdir(input_dir):
    if not file_name.endswith("_GT.npy"):
        continue  # GT 파일이 아닌 경우 무시

    # ✅ 파일명에서 카테고리 추출 (숫자 제거)
    match = re.match(r"([A-Za-z]+)", file_name)  
    if not match:
        print(f"⚠ {file_name}에서 카테고리명을 찾을 수 없음. 건너뜀")
        continue

    category = match.group(1)  # 예: "Abuse", "Arson" 등

    # ✅ GT 데이터 로드
    file_path = os.path.join(input_dir, file_name)
    gt_data = np.load(file_path)

    # ✅ 카테고리별 GT 병합
    if category not in category_gt:
        category_gt[category] = []
    category_gt[category].append(gt_data)

# ✅ 카테고리별 GT 파일 저장
for category, gt_list in category_gt.items():
    merged_gt = np.concatenate(gt_list, axis=0)
    save_path = os.path.join(output_dir, f"{category}_GT.npy")
    np.save(save_path, merged_gt)
    print(f"📂 Saved {category} GT data to {save_path} (shape: {merged_gt.shape})")

print("✅ GT 데이터 **카테고리별로 병합** 및 저장 완료!")
