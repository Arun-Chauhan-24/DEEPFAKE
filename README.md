---

# 👁️ DeepSight — ISTVT Video Deepfake Detector

DeepSight is a Streamlit web application designed to analyze videos for deepfake manipulation. It is powered by **ISTVT (Interpretable Spatial-Temporal Video Transformer)**, a custom PyTorch model that pairs an Xception CNN backbone with decomposed spatial and temporal attention blocks to detect anomalies in video frames.

**🔗 Live Demo:** [Launch DeepSight](https://deepfake-huycynwhmndahtyxc4dczd.streamlit.app/)

---

## ✨ Key Features

* **Browser-Friendly Analysis:** Directly upload and analyze `.mp4`, `.avi`, or `.mov` videos through an intuitive web interface.
* **Memory-Safe Processing:** Automatically samples up to 520 frames from a video, processing them efficiently in chunks of 16 to prevent memory overloads.
* **Smart Scoring (Top-10% Pooling):** The final anomaly score is based on the average of the most suspicious 10% of frame-windows. This prevents long stretches of normal footage from hiding a brief deepfake segment.
* **Calibrated Confidence:** Uses an exponential curve to sharpen confidence scores on the "authentic" side of the threshold for a more accurate final verdict.
* **Visual Proof:** Generates an attention heatmap over the most suspicious frame, showing exactly *where* the model detected manipulation.
* **Plug-and-Play:** Automatically downloads and caches the required model weights from Kaggle on the first run—no manual setup required.

---

## 🧠 How It Works

DeepSight breaks down video analysis into three core components:

### 1. Model Architecture (`model_utils.py`)

* **Backbone:** An `xception` model (via `timm`) extracts spatial features (shapes and textures) from every single frame.
* **SelfSubtract:** Calculates frame-to-frame differences to highlight unnatural changes over time.
* **DecomposedAttentionBlock:** Analyzes temporal patterns (across frames) and spatial patterns (across frame patches) separately, merging them through a Multi-Layer Perceptron (MLP).
* **ISTVT:** Stacks multiple attention blocks on the backbone features, ultimately generating a single "real vs. fake" probability for every 4-frame window.

### 2. Streaming Inference Pipeline (`predict_video`)

1. **Sampling:** The video is sampled at an even stride up to a maximum of 520 frames (rounded to a multiple of 4).
2. **Chunking:** Frames are processed in small chunks of 16 to maintain low memory usage.
3. **Windowing:** Each chunk is divided into 4-frame windows. The model scores each window.
4. **Pooling:** The top 10% highest anomaly scores are averaged to generate the final video-level verdict.
5. **Heatmap Generation:** A forward hook on the model's projection layer grabs the attention map from the most suspicious window and creates a visual overlay.

### 3. Verdict Calibration (`app.py`)

* **Deepfake (Score > 0.5):** The confidence percentage scales linearly with the raw probability.
* **Authentic (Score ≤ 0.5):** To counter the naturally pessimistic baseline caused by Top-10% pooling, confidence is boosted using an exponential curve: `100 − 50 × (probability / 0.5)³`.

---

## 📂 Repository Structure

```text
.
├── app.py              # Streamlit UI, model caching, video upload & UI logic
├── model_utils.py      # ISTVT architecture, streaming inference engine, heatmaps
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

```

---

## ⚙️ Installation

To run this project locally, clone the repository and install the required dependencies:

```bash
git clone https://github.com/xenoz27/DeepFake.git
cd DeepFake
pip install -r requirements.txt

```

### Dependencies

* `streamlit`
* `kagglehub`
* `torch` & `torchvision`
* `opencv-python-headless`
* `numpy`
* `Pillow`
* `timm`

---

## 🚀 Usage

Start the Streamlit app locally:

```bash
streamlit run app.py

```

**First Launch Note:** The app will use `kagglehub` to automatically download the pretrained ISTVT weights (`istvt_master_weights.pth`) from Kaggle and cache them in memory.

1. Open the provided local URL in your browser (usually `http://localhost:8501`).
2. Upload a supported video file (`.mp4`, `.avi`, or `.mov`).
3. Click **"Run Autonomous ISTVT Analysis"**.
4. Review the final verdict, confidence score, and the generated attention heatmap.

---

## 📊 Dataset & Model Weights

The pre-trained weights for the ISTVT model are hosted publicly on Kaggle. They are fetched automatically at runtime, but can be viewed here:
🔗 **[Kaggle Dataset: istvt-pth](https://www.kaggle.com/datasets/gam888i/istvt-pth)**

---

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| **Web Interface** | Streamlit |
| **Deep Learning** | PyTorch, timm (Xception backbone) |
| **Media Processing** | OpenCV, Pillow, torchvision |
| **Weight Hosting** | Kaggle (via kagglehub) |

---

> **⚠️ Disclaimer**
> *This tool is built for research and educational purposes. The deepfake detection scores are probabilistic estimates and should not be used as definitive, legal proof of a video's authenticity or manipulation.*
