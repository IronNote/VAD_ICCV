import numpy as np
import pandas as pd
import os

clip_len = 16  # ✅ 16 프레임 단위

# 📂 테스트 리스트 및 GT 파일 경로
feature_list = 'C:/Myoung_Chul_KIM/VadCLIP(AAAI_2024)/VadCLIP-main/list/ucf_CLIP_rgbtest_category.csv'  # ✅ 테스트 리스트 (CSV)
gt_txt = 'C:/Myoung_Chul_KIM/VadCLIP(AAAI_2024)/VadCLIP-main/list/Temporal_Anomaly_Annotation.txt'  # ✅ GT 어노테이션 (TXT)
output_dir = "C:/Myoung_Chul_KIM/VadCLIP(AAAI_2024)/VadCLIP-main/list/category_GT"  # ✅ 저장 폴더

# 📌 GT 어노테이션 읽기
gt_lines = list(open(gt_txt))

# 📌 카테고리별 GT 저장 딕셔너리
category_gt = {}

# ✅ CSV 파일 읽기
lists = pd.read_csv(feature_list)

for idx in range(lists.shape[0]):
    file_path = lists.loc[idx]['path']

    # ✅ '__5.npy' 파일만 처리
    if '__5.npy' not in file_path:
        continue

    # ✅ npy 파일명에서 카테고리 추출
    file_name = os.path.basename(file_path)  # Abuse028_x264__5.npy
    video_base_name = file_name.split("__")[0]  # Abuse028_x264
    category = video_base_name.split("_")[0]  # Abuse

    # ✅ Normal 데이터 제외
    if category == "Normal":
        continue

    # ✅ feature 파일 로드 및 클립 길이 계산
    fea = np.load(file_path)
    num_frames = (fea.shape[0] + 1) * clip_len  # 클립 개수 * 16프레임

    # ✅ GT 벡터 초기화
    gt_vec = np.zeros(num_frames).astype(np.float32)

    # ✅ GT 어노테이션 매칭
    for gt_line in gt_lines:
        if video_base_name in gt_line:
            gt_content = gt_line.strip('\n').split('  ')[1:-1]
            abnormal_fragment = [[int(gt_content[i]), int(gt_content[j])]
                                 for i in range(1, len(gt_content), 2)
                                 for j in range(2, len(gt_content), 2) if j == i + 1]
            if abnormal_fragment:
                abnormal_fragment = np.array(abnormal_fragment)
                for frag in abnormal_fragment:
                    if frag[0] != -1 and frag[1] != -1:
                        gt_vec[frag[0]:frag[1]] = 1.0
            break

    # ✅ 카테고리별 GT 저장
    if category not in category_gt:
        category_gt[category] = []
    category_gt[category].append(gt_vec[:-clip_len])  # 마지막 clip_len 만큼 제외

# ✅ 카테고리별 GT 저장
os.makedirs(output_dir, exist_ok=True)

for category, gt_list in category_gt.items():
    gt_array = np.concatenate(gt_list, axis=0)
    save_path = os.path.join(output_dir, f"{category}_GT.npy")
    np.save(save_path, gt_array)
    print(f"📂 Saved {category} GT data to {save_path} (shape: {gt_array.shape})")

print("✅ GT 데이터 카테고리별로 분할 및 저장 완료!")