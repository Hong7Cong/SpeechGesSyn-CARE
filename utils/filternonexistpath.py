import os
import pandas as pd
import torch
import torchvision
from torchvision.io import read_video
import cv2
from tqdm import tqdm

# =========================
# Configuration
# =========================
INPUT_CSV = "../clips2.csv"
OUTPUT_CSV = "../clips_filtered.csv"

# =========================
# Main
# =========================
# Optional: set resize to match your dataset class
RESIZE = None  # e.g., 224


# =========================
# Video loading attempts
# =========================
def try_read_torchvision(path):
    try:
        frames, _, _ = read_video(path, pts_unit="sec")
        if frames is None or frames.numel() == 0:
            return False
        return True
    except Exception:
        return False


def try_read_opencv(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return False
        return True
    except Exception:
        return False


def is_video_loadable(path):
    if not os.path.exists(path):
        return False

    # First attempt: torchvision
    if try_read_torchvision(path):
        return True

    # Second attempt: OpenCV fallback
    if try_read_opencv(path):
        return True

    return False


# =========================
# Main
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    if "path_to_clip" not in df.columns:
        raise ValueError("CSV must contain 'path_to_clip' column")

    print(f"Total samples: {len(df)}")

    valid_mask = []
    for path in tqdm(df["path_to_clip"]):
        valid_mask.append(is_video_loadable(path))

    df["valid_video"] = valid_mask

    filtered_df = df[df["valid_video"]].drop(columns=["valid_video"])

    print(f"Valid videos: {len(filtered_df)}")
    print(f"Removed videos: {len(df) - len(filtered_df)}")

    filtered_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned CSV to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()