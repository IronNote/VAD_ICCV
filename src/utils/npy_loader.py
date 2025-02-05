import numpy as np
import os

# 특정 디렉토리에 있는 모든 npy 파일 읽기
directory_path = "C:/Myoung_Chul_KIM/VadCLIP(AAAI_2024)/VadCLIP-main/list"

for filename in os.listdir(directory_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(directory_path, filename)
        data = np.load(file_path, allow_pickle=True) 
        print(f"파일명: {filename}, 데이터 형태: {data.shape}")