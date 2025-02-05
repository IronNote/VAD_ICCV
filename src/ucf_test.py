import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import CLIPProcessor, CLIPModel
import os
import glob
import re
import ast

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option

# **********************************************************************************************************
# 📌 원본 비디오 경로 찾기 (카테고리 자동 추출)
# **********************************************************************************************************

def get_original_video_path(video_root_folder, base_name):
    """
    주어진 비디오 파일명에서 카테고리를 추출하고, 해당 카테고리 내에서 원본 비디오 파일 경로를 찾기
    """
    base_name_cleaned = re.sub(r"__\d+$", "", base_name)  # "__숫자" 제거
    category = base_name_cleaned.split("0")[0]  # 첫 숫자 이전까지가 카테고리

    video_folder = os.path.join(video_root_folder, category)
    video_candidates = glob.glob(os.path.join(video_folder, f"{base_name_cleaned}*"))

    if len(video_candidates) == 0:
        print(f"❌ Error: No matching video found for {base_name_cleaned} in {video_folder}")
        return None, category
    elif len(video_candidates) > 1:
        print(f"⚠ Warning: Multiple matching videos found for {base_name_cleaned}, selecting the first one.")

    video_path = video_candidates[0]
    print(f"✅ Original video path: {video_path}")
    return video_path, category

# **********************************************************************************************************
# 📌 모델 테스트 수행 (AUC, AP, mAP 평가)
# **********************************************************************************************************

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    """
    📌 test 함수: 모델의 성능을 평가
    - AUC, AP, mAP를 통해 anomaly detection 성능 측정
    - testdataloader에서 데이터를 불러와 모델 추론 후 성능 분석
    """

    model.to(device)
    model.eval()

    element_logits2_stack = []

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])

            if length < maxlen:
                visual = visual.unsqueeze(0)
            
            visual = visual.to(device)

            # 📊 비디오 데이터를 segment 단위로 나눔
            lengths = torch.zeros(int(length / maxlen) + 1, dtype=int)
            for j in range(len(lengths)):
                lengths[j] = min(maxlen, length)
                length -= maxlen
            
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            
            # 🎯 모델 추론
            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])
            
            # 🟢 이상 행동 확률 계산
            prob1 = torch.sigmoid(logits1[:length].squeeze(-1))
            prob2 = (1 - logits2[:length].softmax(dim=-1)[:, 0].squeeze(-1)) # 해당 결과가 어떤 의미? 차원()

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[:length].softmax(dim=-1).cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, axis=0)
            element_logits2_stack.append(element_logits2)

    # AUC, AP 성능 평가
    ap1 = np.repeat(ap1.cpu().numpy(), 16)
    ap2 = np.repeat(ap2.cpu().numpy(), 16)

    ROC1 = roc_auc_score(gt, ap1)
    AP1 = average_precision_score(gt, ap1)
    ROC2 = roc_auc_score(gt, ap2)
    AP2 = average_precision_score(gt, ap2)

    print("📊 AUC1:", ROC1, " AP1:", AP1)
    print("📊 AUC2:", ROC2, " AP2:", AP2)

    # Mean Average Precision(mAP) 평가
    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    avg_map = np.mean(dmap)

    for i, val in enumerate(dmap):
        print(f"📊 mAP@{iou[i]:.1f} = {val:.2f}%")
    print(f"📊 평균 mAP: {avg_map:.2f}%\n")

    return ROC1, AP1

# **********************************************************************************************************
# 📌 카테고리별 테스트 수행
# **********************************************************************************************************

def plot_category_performance(category_results, gt_dir):
    """
    📊 카테고리별 AUC, AP 성능을 시각화하여 저장하는 함수
    """
    categories = list(category_results.keys())
    auc1_scores = [category_results[cat]["AUC1"] for cat in categories]
    ap1_scores = [category_results[cat]["AP1"] for cat in categories]
    auc2_scores = [category_results[cat]["AUC2"] for cat in categories]
    ap2_scores = [category_results[cat]["AP2"] for cat in categories]

    x = range(len(categories))

    plt.figure(figsize=(15, 6))
    bars1 = plt.bar(x, auc1_scores, width=0.2, label="AUC1", color="blue", alpha=0.7)
    plt.bar([i + 0.2 for i in x], ap1_scores, width=0.2, label="AP1", color="green", alpha=0.7)
    plt.bar([i + 0.4 for i in x], auc2_scores, width=0.2, label="AUC2", color="red", alpha=0.7)
    plt.bar([i + 0.6 for i in x], ap2_scores, width=0.2, label="AP2", color="orange", alpha=0.7)

    plt.xticks([i + 0.3 for i in x], categories, rotation=90)
    plt.ylabel("Score")
    plt.title("Category-wise Anomaly Detection Performance")
    plt.legend()

    # ✅ AUC1 막대 위에 수치 표시
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha="center", fontsize=10, fontweight="bold")

    save_path = os.path.join(gt_dir, "category_performance.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"📂 Saved performance plot to {save_path}")

def test_category_wise(model, gt_dir, maxlen, prompt_text, label_map, device):
    """
    📌 카테고리별 anomaly detection 성능 평가 (Frame-Level)
    """

    model.to(device)
    model.eval()

    category_results = {}

    category_gt_files = [f for f in os.listdir(gt_dir) if f.endswith("_GT.npy")]
    category_names = [f.replace("_GT.npy", "") for f in category_gt_files]

    for category in category_names:
        print(f"\n🚀 Evaluating {category} category...\n")

        category_test_list = os.path.join(gt_dir, f"{category}_category_test.csv")
        if not os.path.exists(category_test_list):
            print(f"⚠ {category}_category_test.csv 파일이 없음 - 건너뜀")
            continue

        testdataset = UCFDataset(maxlen, category_test_list, True, label_map)  
        testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

        category_gt = np.load(os.path.join(gt_dir, f"{category}_GT.npy"))
        print(f"📂 Loaded GT for {category} (shape: {category_gt.shape})")

        ap1_list = []
        ap2_list = []

        with torch.no_grad():
            for i, item in enumerate(testdataloader):
                visual = item[0].squeeze(0)
                length = int(item[2])

                if length < maxlen:
                    visual = visual.unsqueeze(0)

                visual = visual.to(device)

                lengths = torch.zeros(int(length / maxlen) + 1, dtype=int)
                for j in range(len(lengths)):
                    lengths[j] = min(maxlen, length)
                    length -= maxlen

                padding_mask = get_batch_mask(lengths, maxlen).to(device)

                _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
                logits1 = logits1.reshape(-1, logits1.shape[2])
                logits2 = logits2.reshape(-1, logits2.shape[2])

                prob1 = torch.sigmoid(logits1[:length].squeeze(-1)).cpu().numpy()
                prob2 = (1 - logits2[:length].softmax(dim=-1)[:, 0].squeeze(-1)).cpu().numpy()
                
                ap1_list.append(prob1)
                ap2_list.append(prob2)

        ap1 = np.concatenate(ap1_list, axis=0)
        ap2 = np.concatenate(ap2_list, axis=0)
        ap1 = np.repeat(ap1, 16)
        ap2 = np.repeat(ap2, 16)

        print(f"\n📌 [Category: {category}] GT vs AP 크기 비교")
        print(f"✅ GT 크기: {len(category_gt)}, AP1 크기: {len(ap1)}")

        unique_gt_values = np.unique(category_gt)
        print(f"🛠 [DEBUG] {category} GT unique values: {unique_gt_values}")

        if len(unique_gt_values) < 2:
            print(f"⚠ [주의] {category} GT 데이터가 단일 클래스임 - AUC 계산 제외")
            AP1 = average_precision_score(category_gt, ap1)
            print(f"📊 AP1: {AP1:.4f}")
            category_results[category] = {"AUC1": 0, "AP1": AP1, "AUC2": 0, "AP2": 0}
            continue
        
        ROC1 = roc_auc_score(category_gt, ap1)
        AP1 = average_precision_score(category_gt, ap1)
        ROC2 = roc_auc_score(category_gt, ap2)
        AP2 = average_precision_score(category_gt, ap2)

        print(f"📊 AUC1: {ROC1:.4f}, AP1: {AP1:.4f}")
        print(f"📊 AUC2: {ROC2:.4f}, AP2: {AP2:.4f}")

        category_results[category] = {"AUC1": ROC1, "AP1": AP1, "AUC2": ROC2, "AP2": AP2}

    print("\n🚀 모든 카테고리 평가 완료\n")

    # 📊 성능 그래프 저장
    plot_category_performance(category_results, gt_dir)

# **********************************************************************************************************
# 📊 카테고리별 GT 데이터 분포 확인
# **********************************************************************************************************

def analyze_category_gt_distribution(gt_dir, save_dir):
    """
    📌 Analyze and visualize frame-level distribution for each category.
    - Counts total frames, anomaly (1) frames, and normal (0) frames.
    - Saves distribution graph in the given directory.
    """

    # ✅ Load category-wise GT files
    category_gt_files = [f for f in os.listdir(gt_dir) if f.endswith("_GT.npy")]
    category_names = [f.replace("_GT.npy", "") for f in category_gt_files]

    # ✅ Dictionaries to store frame counts
    category_frame_counts = {}  # Total frames per category
    category_anomaly_counts = {}  # Anomaly (1) frames per category
    category_normal_counts = {}  # Normal (0) frames per category

    for category in category_names:
        gt_path = os.path.join(gt_dir, f"{category}_GT.npy")
        category_gt = np.load(gt_path)

        total_frames = len(category_gt)  # Total frames
        anomaly_frames = np.sum(category_gt == 1)  # Anomaly frames
        normal_frames = np.sum(category_gt == 0)  # Normal frames

        category_frame_counts[category] = total_frames
        category_anomaly_counts[category] = anomaly_frames
        category_normal_counts[category] = normal_frames

        print(f"📌 {category} - Total frames: {total_frames}, Anomaly frames: {anomaly_frames}, Normal frames: {normal_frames}")

    # ✅ Plot frame distribution (Bar Chart)
    categories = list(category_frame_counts.keys())
    total_values = [category_frame_counts[c] for c in categories]
    anomaly_values = [category_anomaly_counts[c] for c in categories]
    normal_values = [category_normal_counts[c] for c in categories]

    x = np.arange(len(categories))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, total_values, width=0.3, label="Total Frames", color="blue")
    plt.bar(x, anomaly_values, width=0.3, label="Anomaly Frames (1)", color="red")
    plt.bar(x + 0.2, normal_values, width=0.3, label="Normal Frames (0)", color="green")

    plt.xticks(x, categories, rotation=45, ha="right")
    plt.ylabel("Frame Count")
    plt.title("Frame Distribution per Category")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # ✅ Save the graph
    save_path = os.path.join(save_dir, "category_GT_distribution.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"\n📂 Distribution graph saved: {save_path}")

    plt.show()

# **********************************************************************************************************
# 📊 점수 그래프 비디오 파일 저장 & 애니메이션 함수 수행
# **********************************************************************************************************

def clean_video_name(video_name):
    """
    📌 비디오 이름에서 '_x264' 이후의 내용을 제거하여 GT와 일치시키는 함수
    예: 'Abuse028_x264__5' -> 'Abuse028_x264'
    """
    match = re.match(r"(.+_x264)", video_name)
    return match.group(1) if match else video_name  # `_x264`까지만 유지

def load_gt_from_annotation(annotation_path):
    """
    📌 Temporal_Anomaly_Annotation.txt에서 GT 정보를 불러오는 함수
    """
    gt_dict = {}
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            video_name = parts[0]  # 첫 번째 열이 비디오 파일명
            anomaly_frames = list(map(int, parts[2:]))  # 시작~끝 프레임 정보

            if video_name not in gt_dict:
                gt_dict[video_name] = []

            # GT에 정상적인 이상 구간만 추가
            for i in range(0, len(anomaly_frames), 2):
                if anomaly_frames[i] != -1:
                    gt_dict[video_name].append((anomaly_frames[i], anomaly_frames[i+1]))

    return gt_dict

def real_time_scoring_animation_with_gt(model, input_dataloader, maxlen, prompt_text, device, output_root, annotation_path, fps=30):
    """
    📌 실시간 Scoring + GT 비교 애니메이션 생성 (Frame-level GT 반영)
    - GT는 Temporal_Anomaly_Annotation.txt에서 로드하여 사용
    """

    model.to(device)
    model.eval()
    video_root_folder = "D:/UCF_Crimes/Videos"

    # 🔥 GT 데이터 로드 (이름 정리 후 매칭)
    gt_dict = load_gt_from_annotation(annotation_path)

    with torch.no_grad():
        for i, item in enumerate(input_dataloader):
            visual = item[0].squeeze(0)
            length = item[2].item()
            file_name = item[3][0]
            base_name = os.path.splitext(file_name)[0]
            clean_base_name = clean_video_name(base_name)  # 🛠 이름 정리

            print(f"\n📌 Processing: {base_name} -> {clean_base_name}")
            print(f"📊 visual.shape: {visual.shape}, length: {length}")

            # 원본 비디오 찾기 & 카테고리명 추출
            video_path, category = get_original_video_path(video_root_folder, clean_base_name)
            if video_path is None:
                print(f"❌ Error: No matching video found for {clean_base_name} in {video_root_folder}")
                continue  # 비디오가 없으면 넘어감

            # 📌 GT 데이터 로드 (frame 단위)
            if clean_base_name not in gt_dict:
                print(f"⚠ GT 정보 없음: {clean_base_name}")
                continue
            anomaly_regions = gt_dict[clean_base_name]
            print(f"✅ GT Loaded: {clean_base_name} -> {anomaly_regions}")

            # 비디오 정보 가져오기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"❌ Unable to open video file: {video_path}")

            video_fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            print(f"🎬 원본 비디오 정보: FPS = {video_fps}, 총 프레임 = {total_frames}")

            # 카테고리별 폴더 생성 (없으면 자동 생성)
            category_folder = os.path.join(output_root, category)
            os.makedirs(category_folder, exist_ok=True)

            # Feature Chunks 생성
            num_chunks = int(np.ceil(length / maxlen))
            features_chunks = []
            for j in range(num_chunks):
                start_idx = j * maxlen
                end_idx = min((j + 1) * maxlen, length)
                chunk = visual[start_idx:end_idx]

                if chunk.shape[0] < maxlen:
                    padding = torch.zeros((maxlen - chunk.shape[0], visual.shape[1]), device=device)
                    chunk = torch.cat([chunk, padding], dim=0)

                features_chunks.append(chunk)

            print(f"📊 Total Segments in npy: {length // 16}, Expected Chunks: {num_chunks}")

            if len(features_chunks) == 0:
                print(f"⚠ Warning: No features_chunks created for {base_name}. Skipping...")
                continue

            # 🟢 Model Inference & Score Calculation
            scores = []
            for chunk in features_chunks:
                chunk_tensor = chunk.unsqueeze(0).to(device)
                padding_mask = get_batch_mask(torch.tensor([maxlen]), maxlen).to(device)

                _, logits1, _ = model(chunk_tensor, padding_mask, prompt_text, lengths=[maxlen])
                prob1 = torch.sigmoid(logits1.view(-1)).detach().cpu().numpy()
                scores.extend(prob1)

            print(f"📊 Scores computed: {len(scores)}")

            if len(scores) == 0:
                print(f"⚠ Warning: No scores generated for {base_name}. Skipping...")
                continue

            # 🎥 FPS 맞춰 Score Resampling
            frame_timestamps = np.linspace(0, total_frames / video_fps, total_frames)
            score_timestamps = np.linspace(0, total_frames / video_fps, len(scores))
            scores_resampled = np.interp(frame_timestamps, score_timestamps, scores)

            # ✅ GT 프레임 벡터 생성 (0: 정상, 1: 이상)
            gt_frame_vector = np.zeros(total_frames)
            for start, end in anomaly_regions:
                gt_frame_vector[start:end] = 1  # 이상 행동 프레임 설정

            # 🎬 애니메이션 설정
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.set_xlim(0, total_frames / video_fps)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Score")
            ax.grid()

            line_score, = ax.plot([], [], label="Anomaly Score", color="red")
            line_gt, = ax.plot([], [], label="GT", color="blue", linestyle="dashed")
            scatter = ax.scatter([], [], color="black", s=50)
            ax.legend(loc="upper right")

            def init():
                line_score.set_data([], [])
                line_gt.set_data([], [])
                scatter.set_offsets(np.array([[], []]).T)
                return line_score, line_gt, scatter

            def update(frame_idx):
                x_data = frame_timestamps[:frame_idx + 1]
                y_score_data = scores_resampled[:frame_idx + 1]
                y_gt_data = gt_frame_vector[:frame_idx + 1]

                line_score.set_data(x_data, y_score_data)
                line_gt.set_data(x_data, y_gt_data)
                scatter.set_offsets(np.column_stack([x_data[-1:], y_score_data[-1:]]))

                return line_score, line_gt, scatter

            ani = FuncAnimation(fig, update, frames=len(frame_timestamps), init_func=init, blit=True)

            # 🎥 카테고리별 저장 경로 설정
            output_path = os.path.join(category_folder, f"{base_name}_scores_with_GT.mp4")
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='RealTimeScoring'), bitrate=1800)
            ani.save(output_path, writer=writer)
            print(f"✅ Animation saved to {output_path}")

            plt.close(fig)

# **********************************************************************************************************
# 📌 비디오 영상 입력 -> CLIP Feature Extraction
# **********************************************************************************************************

def extract_clip_features(video_path, clip_model, processor, device):
    """
    비디오의 각 프레임에서 CLIP 특징을 추출하는 함수.
    
    Args:
        video_path (str): 입력 비디오 경로
        clip_model (CLIPModel): CLIP 모델
        processor (CLIPProcessor): CLIP 입력 전처리기
        device (str): 실행 장치 (CPU/GPU)

    Returns:
        np.ndarray: CLIP 특징 벡터 리스트 (프레임 단위)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Unable to open video file: {video_path}")

    features = []
    print(f"📊 Extracting CLIP features for video: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 📏 프레임을 CLIP 입력 크기로 조정 및 전처리
        frame_resized = cv2.resize(frame, (224, 224))  
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        inputs = processor(images=frame_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            feature = clip_model.get_image_features(**inputs).cpu().numpy()
        features.append(feature)

    cap.release()
    features = np.vstack(features)  # 전체 특징 벡터 스택
    print(f"✅ Extracted features shape: {features.shape}")
    return features

# **********************************************************************************************************
# 📌 비디오 입력 -> CLIP 특징 추출 -> 이상치 점수 계산 -> anomaly_video_score_plotting 함수 호출
# **********************************************************************************************************

def test_real_time_vadclip(model, video_path, visual_length, prompt_text, device):
    """
    실시간 비디오 분석을 수행하여 이상 행동 탐지를 진행하는 함수.

    Args:
        model (CLIPVAD): 이상 행동 탐지 모델
        video_path (str): 입력 비디오 경로
        visual_length (int): 모델 입력에 사용할 시퀀스 길이
        prompt_text (str): CLIP 텍스트 프롬프트
        device (str): 실행 장치 (CPU/GPU)

    Returns:
        None
    """
    model.eval()
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Unable to open video file: {video_path}")

    features = []
    print(f"📊 Extracting CLIP features for video: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 📏 프레임을 CLIP 입력 크기로 변환
        resized_frame = cv2.resize(frame, (224, 224)) / 255.0
        resized_frame = np.transpose(resized_frame, (2, 0, 1))  # HWC -> CHW 변환
        resized_frame = torch.tensor(resized_frame, dtype=torch.float32).unsqueeze(0).to(device)

        # 🎯 CLIP 특징 추출
        with torch.no_grad():
            clip_features = model.clipmodel.encode_image(resized_frame).cpu().numpy()
        features.append(clip_features)

    cap.release()
    features = np.vstack(features)
    print(f"✅ Extracted features shape: {features.shape}")

    # 📌 특징을 visual_length 단위로 분할하여 입력
    num_chunks = int(np.ceil(features.shape[0] / visual_length))
    features_chunks = []

    for i in range(num_chunks):
        start_idx = i * visual_length
        end_idx = min((i + 1) * visual_length, features.shape[0])
        chunk = features[start_idx:end_idx]

        # 🏳️ 부족한 부분은 0으로 패딩
        if chunk.shape[0] < visual_length:
            padding = np.zeros((visual_length - chunk.shape[0], features.shape[1]), dtype=np.float32)
            chunk = np.vstack([chunk, padding])

        features_chunks.append(chunk)

    # 🔍 모델을 통해 이상 행동 점수 계산
    scores = []
    for chunk in features_chunks:
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        padding_mask = get_batch_mask(torch.tensor([visual_length]), visual_length).to(device)

        _, logits1, _ = model(chunk_tensor, padding_mask, prompt_text, lengths=[visual_length])
        prob1 = torch.sigmoid(logits1.view(-1)).detach().cpu().numpy()

        scores.extend(prob1)

    # 📊 점수를 기반으로 비디오 시각화 수행
    visualize_video_anomaly_scores(
        video_path=video_path,
        scores=scores,
        output_path="anomaly_scores.mp4"
    )


# **********************************************************************************************************
# 📊 스코어를 OpenCV 그래프로 플로팅하여 이상 행동 시각화
# **********************************************************************************************************

def visualize_video_anomaly_scores(video_path, scores, output_path):
    """
    이상 행동 점수를 기반으로 비디오를 시각화하고 저장하는 함수.

    Args:
        video_path (str): 입력 비디오 경로
        scores (list): 이상 행동 점수 리스트
        output_path (str): 저장할 비디오 파일 경로

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Unable to open video file: {video_path}")

    # 🎥 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 200))  

    frame_idx = 0
    fig, ax = plt.subplots(figsize=(10, 2))  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(scores):
            break

        # 🔴 점수에 따라 프레임에 이상 탐지 결과 표시
        anomaly_score = scores[frame_idx]
        frame = cv2.putText(
            frame,
            f"Anomaly Score: {anomaly_score:.4f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if anomaly_score > 0.5 else (0, 255, 0),  # 이상 점수 0.5 이상일 경우 빨간색
            2
        )

        # 📊 점수 그래프 플로팅
        ax.clear()
        ax.plot(range(frame_idx + 1), scores[:frame_idx + 1], label="Anomaly Score", color="red")
        ax.set_xlim(0, len(scores))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Score")
        ax.legend(loc="upper right")
        ax.grid()

        # 📷 그래프를 이미지로 변환
        fig.canvas.draw()
        graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)

        # 🎬 프레임과 그래프를 합성
        combined_frame = np.zeros((height + 200, width, 3), dtype=np.uint8)
        combined_frame[:height, :, :] = frame
        graph_resized = cv2.resize(graph_img, (width, 200))
        combined_frame[height:, :, :] = graph_resized

        # 🎥 비디오 저장
        out.write(combined_frame)

        # 🔎 실시간 표시
        cv2.imshow("Anomaly Detection", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video with anomaly scores saved to {output_path}")

# **********************************************************************************************************
# 📌 메인 실행
# **********************************************************************************************************

if __name__ == '__main__':
    # 🔥 장치 설정 (CUDA 사용 가능 시 GPU 활용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    # 🎯 프롬프트 텍스트 (레이블 매핑 설정)
    label_map = dict({
        'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault',
        'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents',
        'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
        'Vandalism': 'Vandalism'
    })

    # 📝 프롬프트 텍스트 생성
    prompt_text = get_prompt_text(label_map)

    # 🔧 모델 초기화 및 로드
    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head,
        args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, device
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 🖼️ CLIP 모델 및 프로세서 로드
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # **********************************************************************************************************
    # 🎯 UCF CRIME 성능 테스트
    if args.test_mode == 'test':
        print("🚀 Running in test mode...")
        
        # 📂 데이터셋 로드
        testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map)  
        testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)  

        # 📊 Ground Truth 데이터 로드
        gt = np.load(args.gt_path) 
        gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
        gtlabels = np.load(args.gt_label_path, allow_pickle=True)

        # 🔍 모델 평가 수행
        test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
    
    # **********************************************************************************************************
    # 🎯 UCF CRIME 카테고리별 성능 테스트
    if args.test_mode == 'Category_test':
        print("🚀 Running in Category test mode...")
        
        # 🔍 모델 평가 수행
        test_category_wise(model, args.category_gt_dir, args.visual_length, prompt_text, label_map, device)
    
    # **********************************************************************************************************
    # 📈 UCF CRIME 카테고리별 데이터 분포 확인
    if args.test_mode == 'gt_distribution':
        print("🚀 Running in analyze category gt distribution mode...")
        
        # 🔍 모델 평가 수행
        analyze_category_gt_distribution(args.category_gt_dir, args.category_gt_dir)

    # **********************************************************************************************************
    # 📈 이상치 점수 그래프 생성 모드
    if args.test_mode == 'plot_score':
        print("📊 Running to plot score...")

        # 📂 데이터셋 로드
        input_dataset = UCFDataset(args.visual_length, args.test_list, False, label_map)
        input_dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False)

        # 🎬 실시간 이상치 점수 애니메이션 생성
        real_time_scoring_animation_with_gt(
            model=model,
            input_dataloader=input_dataloader,
            maxlen=args.visual_length,
            prompt_text=prompt_text,
            device=device,
            output_root=args.score_video_save_path,
            annotation_path=args.annotation_path,
            fps=30
        )

    # **********************************************************************************************************
    # 🎥 실시간 비디오 이상 행동 탐지 모드
    if args.test_mode == 'test_real_time':
        print("🎥 Running in test_real_time mode...")

        # 🔎 실시간 VAD (Video Anomaly Detection) 실행
        test_real_time_vadclip(
            model=model,
            video_path=args.video_path,
            prompt_text=prompt_text,
            device=device,
            visual_length=args.visual_length
        )
    
