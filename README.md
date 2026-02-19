🧠 EEG Stress Detection System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/AniketAgarwal-aa/eeg-stress-detection?style=social)](https://github.com/AniketAgarwal-aa/eeg-stress-detection)

**Research-grade EEG-based stress classification system** using SEED dataset with classical ML and deep learning pipelines.

---

📋 Overview

This system detects stress levels (Low/Medium/High) from EEG signals using:
- **SEED Dataset** (62-channel EEG, 15 subjects)
- **Classical ML**: Random Forest, XGBoost, SVM, etc.
- **Deep Learning**: EEGNet architecture optimized for CPU/GPU
- **Cross-validation**: Leave-One-Subject-Out (LOSO) for generalization

---

🏗️ Architecture
Raw EEG → Feature Extraction → Classical ML → Stress Level
↘ Raw Windows → EEGNet → Stress Level

---

## 📊 Current Results

| Model             | Random Split | LOSO (Cross-Subject) |
|-------------------|--------------|----------------------|
| Random Forest     | 70-75%       | 35-40%               |
| XGBoost           | 72-77%       | 36-42%               |
| **EEGNet (GPU)**  | **85-90%**   | **55-65%**           |

---

## 🚀 Quick Start

### 1️⃣ Setup Environment
# Clone repository
git clone https://github.com/AniketAgarwal-aa/eeg-stress-detection.git
cd eeg-stress-detection

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
2️⃣ Add Dataset
Place SEED dataset in:

text
data/raw/seed/Preprocessed_EEG/
3️⃣ Run Classical ML
bash
python -m src.models.classical.benchmark
4️⃣ Run Deep Learning (Recommended on Colab)
Open notebooks/colab_setup.ipynb in Google Colab with GPU.

📁 Project Structure
text
eeg-stress-detection/
├── configs/          # Configuration files
├── data/             # Dataset (ignored by git)
├── src/              # Source code
│   ├── data/         # Data loaders
│   ├── features/     # Feature extraction
│   ├── models/       # ML/DL models
│   ├── training/     # Training pipelines
│   └── utils/        # Utilities
├── outputs/          # Results (ignored)
└── notebooks/        # Colab notebooks
🧠 Key Features
✅ Memory-efficient streaming dataset (works with 8GB RAM)

✅ Complete classical ML benchmark (10+ algorithms)

✅ EEGNet implementation optimized for CPU/GPU

✅ LOSO cross-validation for real-world generalization

✅ Colab-ready for GPU training

✅ GitHub-ready with proper .gitignore

📈 Future Work
Hybrid ensemble (classical + deep learning)

WESAD dataset integration

Raspberry Pi deployment

Real-time inference pipeline

📚 Citation
If you use this code in your research, please cite:

bibtex
@misc{agarwal2026eegstress,
  author = {Agarwal, Aniket},
  title = {EEG Stress Detection System},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/AniketAgarwal-aa/eeg-stress-detection}
}
📄 License
MIT License - see LICENSE file for details.