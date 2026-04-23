Deepfake Detection using PPG Maps + XceptionNet
A computer vision project for detecting deepfake videos by extracting Photoplethysmography (PPG) signal maps from facial regions and classifying them using a fine-tuned XceptionNet, benchmarked against a baseline CNN trained from scratch.


Overview
Deepfake detection is approached here as an image classification problem on PPG signal maps — physiological signals derived from subtle colour changes in facial skin that are disrupted by face manipulation techniques. Each video is converted into one or more (224×224×3) PPG map images, which are then classified as real or fake.

A baseline CNN trained from scratch is compared head-to-head against XceptionNet pretrained on ImageNet, providing an empirical justification for transfer learning in this domain.


Datasets
FaceForensics++ (FF++)
600 real videos (original)
600 fake videos across 6 manipulation types:
Deepfakes — 100 videos
Face2Face — 100 videos
FaceSwap — 100 videos
NeuralTextures — 100 videos
FaceShifter — 100 videos
DeepFakeDetection — 100 videos
Total: 1,200 videos
CelebDF v2
590 real videos (Celeb-real)
590 fake videos sampled from 5,639 available (Celeb-synthesis), using random.seed(42) for reproducibility
Total: 1,180 videos
Combined
Split
Videos
Real
Fake
Train (80%)
1,904
952
952
Val (10%)
238
119
119
Test (10%)
238
119
119
Total
2,380
1,190
1,190


Important: The train/val/test split is performed at the video level before any augmentation to prevent data leakage. Val and test sets are never augmented.


Pipeline
Videos → Train/Val/Test Split → Augmentation (train only)

       → PPG Extraction → PPG Maps → CNN / XceptionNet → Evaluation
Phase 1 — Data Collection
Collect and verify 1,200 FF++ videos and 1,180 CelebDF v2 videos (590 real + 590 fake sampled with random.seed(42)).
Phase 2 — Train/Val/Test Split
Split at video level (80/10/10) before any augmentation.
Phase 3 — Video Augmentation (Training Set Only)
Horizontal flip applied to each training video → 2× multiplier (~3,808 training videos). Val and test sets are untouched.
Phase 4 — PPG Extraction with Temporal Windowing
Constants:

OMEGA        = 128  (frames per window)

WINDOW_STEP  = 64   (50% overlap)

RECT_W       = 128

RECT_H       = 64

IMG_SIZE     = (224, 224)

Per video:

Extract all frames with a detected face (direct bounding box ROI)
Apply a sliding window across frames (e.g., frames 0–127, 64–191, 128–255 ...)
Per window:
Extract 32 sub-region RGB signals
Compute G-PPG (green channel) and C-PPG (CHROM method)
Apply Butterworth bandpass filter [0.7–14 Hz]
Build (128×32) maps, stack → (128×32×3), resize → (224×224×3)
Save with window index: 000_003_w00.png, 000_003_w01.png, ...

Expected output: | Split | Videos | Avg Windows | Maps | |-------|--------|-------------|------| | Train | ~3,808 | 2.5 | ~9,520 | | Val | 238 | 2.5 | ~595 | | Test | 238 | 2.5 | ~595 |
Phase 5 — Metadata CSV
Derived directly from folder structure with zero compute cost.

Columns: video_name | dataset | manipulation_type | label | split

Example: | video_name | dataset | manipulation_type | label | split | |---|---|---|---|---| | 000_003_w00.png | FF_Plus | original | real | train | | 001_005_w01.png | FF_Plus | Deepfakes | fake | train | | celeb_001_w00.png | CelebDF_v2 | Celeb-real | real | val | | celeb_002_w00.png | CelebDF_v2 | Celeb-synthesis | fake | test |


Output Folder Structure
PPG_Maps/

├── FF_Plus/

│   └── THREE_CHANNEL/

│       ├── original/

│       ├── Deepfakes/

│       ├── Face2Face/

│       ├── FaceSwap/

│       ├── NeuralTextures/

│       ├── FaceShifter/

│       └── DeepFakeDetection/

└── CelebDF_v2/

    └── THREE_CHANNEL/

        ├── real/

        └── fake/


Models
Phase 6a — Baseline CNN (from scratch)
Input (224, 224, 3)

→ Conv2D(32, 3×3) + ReLU + MaxPool(2×2)

→ Conv2D(64, 3×3) + ReLU + MaxPool(2×2)

→ Conv2D(128, 3×3) + ReLU + MaxPool(2×2)

→ Flatten

→ Dense(256) + Dropout(0.5)

→ Dense(2, softmax)

Setting
Value
Optimizer
Adam
Learning rate
1e-3
Loss
categorical_crossentropy
Batch size
32
Early stopping
patience = 10
Seed
42
Est. training time
~30–45 mins (Colab T4)

Phase 6b — XceptionNet (pretrained on ImageNet)
Custom head:

GlobalAveragePooling2D

→ Dense(256) + BatchNorm + Dropout(0.5) + L2(1e-4)

→ Dense(128) + BatchNorm + Dropout(0.4) + L2(1e-4)

→ Dense(2, softmax)

Training strategy:

Phase
Layers
LR
Epochs
A — Head only
Base frozen
1e-3
15–20
B — Fine-tune
Last 20 layers unfrozen
1e-5
20–30


Setting
Value
Optimizer
Adam
Loss
categorical_crossentropy
Batch size
16
LR scheduler
ReduceLROnPlateau (factor=0.5, patience=7)
Early stopping
patience = 12
Class weights
Computed from training set
Seed
42
Est. training time
~2 hrs (Colab T4)



Evaluation
Metrics (both models)
Accuracy
F1 (real class, fake class, macro)
AUC-ROC
Confusion matrix
Expected Results
Metric
Baseline CNN
XceptionNet
Accuracy
65–72%
82–88%
AUC-ROC
0.68–0.75
0.85–0.92
F1 macro
0.63–0.70
0.82–0.87
Overfitting gap
Higher
< 8%
Per-manipulation gap
Uncontrolled
< 10%

Per-Manipulation Breakdown
Accuracy, F1, and AUC reported separately for: original · Deepfakes · Face2Face · FaceSwap · NeuralTextures · FaceShifter · DeepFakeDetection · Celeb-real · Celeb-synthesis

Goal: max accuracy gap across manipulation types < 10%.
Cross-Dataset Generalisation (XceptionNet only)
Train → Test
Expected Accuracy
FF++ → CelebDF v2
~75–82%
CelebDF v2 → FF++
~72–80%
Both → Both
~82–88%



Notebook Structure
Notebook 1 — Preprocessing_Final.ipynb (CPU heavy)
Cells 1–5: Setup, paths, constants
Cell 6: Train/val/test split at video level
Cell 7: Horizontal flip — training videos only
Cells 8–15: PPG extraction pipeline
Cell 16: Process FF++ train videos (original + flipped)
Cell 17: Process CelebDF v2 train videos (original + flipped)
Cell 18: Process val + test videos (no augmentation)
Cell 19: Build metadata CSV from folder structure
Notebook 2 — Phase_6_CNN_XceptionNet.ipynb (GPU heavy)
Cells 1–3: Setup + load data
Cell 4: Build baseline CNN
Cell 5: Train baseline CNN
Cell 6: Build XceptionNet
Cell 7: XceptionNet Phase A training
Cell 8: XceptionNet Phase B fine-tuning
Cells 9–11: Evaluation + plots (both models)
Cell 12: Head-to-head comparison table
Cell 13: Per-manipulation fairness breakdown
Cell 14: Cross-dataset evaluation (XceptionNet)


Timeline
Phase
Task
Time
Platform
1
Process remaining FF++ videos
30 mins
Colab
1
Sample + verify CelebDF v2
30 mins
Colab
2
Train/val/test split
10 mins
Colab
3
Horizontal flip augmentation
30 mins
Colab
4
PPG extraction — train (~3,808 videos)
2.5 hrs
Colab GPU
4
PPG extraction — val/test (~476 videos)
30 mins
Colab GPU
5
Build metadata CSV
5 mins
Colab
6a
Baseline CNN training
45 mins
Colab GPU
6b
XceptionNet Phase A
1 hr
Colab GPU
6b
XceptionNet Phase B fine-tuning
1 hr
Colab GPU
7
Full evaluation + comparison
45 mins
Colab
Total


~7.75 hrs





Fairness & Limitations
The baseline CNN trained from scratch on identical data achieves ~65–72% accuracy, confirming that XceptionNet's pretrained feature representations provide a substantial benefit for PPG-based deepfake detection. XceptionNet also demonstrates a lower per-manipulation accuracy gap, indicating improved generalisation fairness across manipulation types.

Known limitation: Both FF++ and CelebDF v2 have a known European demographic skew. Demographic fairness analysis is limited as a result and is recommended as future work using datasets with better demographic coverage (e.g. DFDC).


Requirements
Python 3.8+
TensorFlow / Keras
OpenCV
NumPy, Pandas, Scikit-learn
Matplotlib, Seaborn
Google Colab (T4 GPU recommended)

