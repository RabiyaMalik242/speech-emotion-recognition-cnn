# 🎤 Speech Emotion Recognition using Deep Learning

## 📌 Overview
A deep learning system that recognizes human emotions from speech audio using MFCC features and a Convolutional Neural Network (CNN), trained on the RAVDESS dataset. The system classifies speech into 8 emotion categories and provides a real-time prediction interface using Gradio.

---

## 🎯 Results
| Metric | Score |
|--------|-------|
| Train Accuracy | 99.70% |
| Validation Accuracy | 84.03% |
| Test Accuracy | 83.51% |

---

## 🗂️ Dataset
**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 professional actors (12 male, 12 female)
- 1440 audio files
- 8 emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Download: https://zenodo.org/record/1188976

---

## 🛠️ Tech Stack
| Library | Purpose |
|--------|-------|
| TensorFlow / Keras | Deep learning model |
| Librosa | Audio processing & MFCC extraction |
| Scikit-learn | Preprocessing & evaluation |
| Gradio | Real-time web interface |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualizations |

---

## 🏗️ Project Pipeline
1. **Feature Extraction** — MFCC (40 coefficients, 174 time steps) extracted from each audio file
2. **Data Augmentation** — Noise injection and pitch shifting applied to double dataset to ~2880 samples
3. **Normalization** — StandardScaler applied on training data only to prevent data leakage
4. **Train/Test Split** — 80% training, 20% testing with stratification
5. **Model Training** — 3-layer CNN with BatchNormalization, Dropout, and Early Stopping
6. **Evaluation** — Confusion matrix, classification report, accuracy comparison
7. **Deployment** — Real-time Gradio interface for live audio prediction

---

## 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Input | (40 × 174 × 1) |
| Conv2D + BatchNorm + MaxPooling + Dropout | 32 filters, Dropout(0.25) |
| Conv2D + BatchNorm + MaxPooling + Dropout | 64 filters, Dropout(0.25) |
| Conv2D + BatchNorm + MaxPooling + Dropout | 128 filters, Dropout(0.25) |
| Flatten | — |
| Dense + BatchNorm + Dropout | 256 units, Dropout(0.5) |
| Output Dense | 8 units, softmax |

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/RabiyaMalik242/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install dependencies
```bash
pip install tensorflow librosa scikit-learn gradio matplotlib seaborn numpy pandas
```

### 3. Download the RAVDESS dataset
- Download from: https://zenodo.org/record/1188976
- Extract and place folders inside:

```
Speech_Emotion_Project/
└── Audio_Speech_Actors_Datasets/
    ├── Actor_01/
    ├── Actor_02/
    └── ...
```

### 4. Run the notebook
Open `speech_emotion_recognition.ipynb` in Jupyter Notebook and run all cells from top to bottom.

---

## 🖥️ Gradio Interface
The notebook includes a real-time Gradio interface at the end. After running the final cell, a public link is generated where you can:
- Upload any `.wav` audio file
- Record audio directly from your microphone
- Get the predicted emotion with confidence score instantly

---

## 📊 Visualizations Included
- Training accuracy & loss curves
- Train vs Validation vs Test accuracy bar chart
- Confusion matrix
- Classification report (precision, recall, F1-score per class)

---

## ⚠️ Limitations
- Moderate overfitting (~15% train-val gap) due to small dataset size
- Trained on acted speech — may not generalize perfectly to natural real-world speech

---

## 🔮 Future Work
- Integrate transformer-based pretrained models such as **wav2vec 2.0** for higher accuracy
- Train on larger datasets such as **CREMA-D** or **IEMOCAP**
- Deploy as a persistent full-stack web application

---

## 👩‍💻 Author
**Rabiya**
BS Software Engineering — AI/ML Internship Project

