import os, sys, csv, random
import cv2
import pandas as pd

BASE = os.environ.get("PROJ_ROOT")
VIDEOS = os.path.join(BASE, "archive/top_view/videos")
ANNOS = os.path.join(BASE, "archive/top_view/annotations")
OUT = os.path.join(BASE, "training/soccertrack_yolo_dataset")

CLIPS = [
    "D_20220220_1_0000_0030",
    "D_20220220_1_0300_0330",
    "D_20220220_1_0600_0630",
    "D_20220220_1_0900_0930",
    "D_20220220_1_1200_1230",
    "D_20220220_1_1500_1530",
    "D_20220220_1_1770_1800",
]
STRIDE = 12
TARGET_W, TARGET_H = 1920, 1080
VAL_FRACTION = 0.15
SEED = 42
import sys
if len(sys.argv) > 1:
    CLIPS = sys.argv[1:]
    print("Running subset:", CLIPS)


for split in ["train", "valid"]:
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

rng = random.Random(SEED)
total_written = 0

for clip in CLIPS:
    video_path = os.path.join(VIDEOS, clip + ".mp4")
    csv_path = os.path.join(ANNOS, clip + ".csv")
    if not (os.path.isfile(video_path) and os.path.isfile(csv_path)):
        print(f"SKIP missing files for {clip}")
        continue

    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = list(range(0, min(nb_frames, len(df)), STRIDE))
    clip_written = 0

    for fi in frame_indices:
        csv_frame_num = fi + 1
        if csv_frame_num not in df.index:
            continue
        row = df.loc[csv_frame_num]

        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        resized = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)

        lines = []
        for team in ["0", "1"]:
            for pid in [str(i) for i in range(11)]:
                try:
                    bb_h = row[(team, pid, "bb_height")]
                    bb_l = row[(team, pid, "bb_left")]
                    bb_t = row[(team, pid, "bb_top")]
                    bb_w = row[(team, pid, "bb_width")]
                except KeyError:
                    continue
                if pd.isna(bb_h) or pd.isna(bb_l) or pd.isna(bb_t) or pd.isna(bb_w):
                    continue
                if bb_w <= 0 or bb_h <= 0:
                    continue
                xc = (bb_l + bb_w / 2) / orig_w
                yc = (bb_t + bb_h / 2) / orig_h
                wn = bb_w / orig_w
                hn = bb_h / orig_h
                lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        try:
            bb_h = row[("BALL", "BALL", "bb_height")]
            bb_l = row[("BALL", "BALL", "bb_left")]
            bb_t = row[("BALL", "BALL", "bb_top")]
            bb_w = row[("BALL", "BALL", "bb_width")]
            if not (pd.isna(bb_h) or pd.isna(bb_l) or pd.isna(bb_t) or pd.isna(bb_w)) and bb_w > 0 and bb_h > 0:
                xc = (bb_l + bb_w / 2) / orig_w
                yc = (bb_t + bb_h / 2) / orig_h
                wn = bb_w / orig_w
                hn = bb_h / orig_h
                lines.append(f"1 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        except KeyError:
            pass

        if not lines:
            continue

        split = "valid" if rng.random() < VAL_FRACTION else "train"
        stem = f"{clip}_f{csv_frame_num:04d}"
        img_out = os.path.join(OUT, "images", split, stem + ".jpg")
        lbl_out = os.path.join(OUT, "labels", split, stem + ".txt")
        cv2.imwrite(img_out, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        with open(lbl_out, "w") as f:
            f.write("\n".join(lines) + "\n")

        clip_written += 1
        total_written += 1

    cap.release()
    print(f"{clip}: wrote {clip_written} labeled frames (orig {orig_w}x{orig_h}, {nb_frames} frames)", flush=True)

data_yaml = f"""train: {os.path.join(OUT, 'images', 'train')}
val: {os.path.join(OUT, 'images', 'valid')}
nc: 2
names: ['player', 'ball']
"""
with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write(data_yaml)

print(f"TOTAL written: {total_written}")
print("DONE")
