# Anomaly Detection in Medical Datasets: A Med-GEMMA Inspired Approach

**Project for the Master in Biomedical Engineering (BME) Paris**

This repository contains the code, research, and documentation for my Master's project focused on applying Machine Learning techniques to detect unusual patterns in medical datasets. The primary goal is to enhance clinical insights, potentially leading to early risk identification and optimized healthcare processes. This work is conceptually inspired by the principles of advanced healthcare AI models like Google's Med-GEMMA.

---

## 🎯 Project Goal

To develop, evaluate, and interpret machine learning models capable of identifying statistically significant anomalies in structured medical data (e.g., doctor appointments, clinical analyses).

### Key Objectives

1.  **Research & Understand:** Conduct a thorough literature review of anomaly detection techniques in healthcare.
2.  **Preprocess Data:** Acquire, clean, and engineer features from public, anonymized medical datasets (e.g., MIMIC-IV, PhysioNet).
3.  **Model Development:** Implement and train several ML models, from classical baselines (Isolation Forest) to more advanced approaches (Autoencoders, potentially leveraging transformer embeddings).
4.  **Evaluate & Interpret:** Assess model performance using appropriate metrics and interpret the detected anomalies to determine their potential clinical relevance.
5.  **Document:** Maintain a reproducible and well-documented workflow.

---

## 🛠️ Tech Stack (Planned)

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   **Data Handling:** Pandas, NumPy
    *   **ML / DL:** Scikit-learn, TensorFlow/PyTorch
    *   **Visualization:** Matplotlib, Seaborn, Plotly
    *   **NLP/Embeddings (Optional):** Hugging Face `transformers`
*   **Environment:** Jupyter Notebooks, VS Code, Git

---

## 📂 Repository Structure

```
BME/
│
├── 📄 .gitignore
├── 📄 README.md
│
├── 📁 data/
│   ├── 📁 raw/            # Original, immutable data.
│   ├── 📁 processed/      # Cleaned and preprocessed data.
│   └── 📄 .gitkeep        # Placeholder to keep the directory in git.
│
├── 📁 notebooks/
│   ├── 📁 01_data_exploration.ipynb
│   ├── 📁 02_data_preprocessing.ipynb
│   ├── 📁 03_model_development_baseline.ipynb
│   └── 📁 04_model_development_advanced.ipynb
│
├── 📁 src/
│   ├── 📁 data/
│   │   └── 📄 make_dataset.py       # Scripts to download/generate data.
│   ├── 📁 features/
│   │   └── 📄 build_features.py     # Scripts to create features.
│   ├── 📁 models/
│   │   ├── 📄 train_model.py        # Scripts to train models.
│   │   └── 📄 predict_model.py      # Scripts for inference.
│   └── 📁 visualization/
│       └── 📄 visualize.py          # Scripts for visualizations.
│
├── 📁 reports/
│   ├── 📁 figures/          # Figures and plots for the final report.
│   └── 📄 manuscript.pdf    # Final thesis/report.
│
├── 📄 requirements.txt      # Project dependencies.
└── 📄 setup.py              # To make the project installable (optional but good practice).
```

---

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/BME.git
    cd BME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📈 Project Roadmap

*   **[ ] Phase 1: Foundation & Literature Review**
*   **[ ] Phase 2: Data Acquisition & Preprocessing**
*   **[ ] Phase 3: Model Development & Experimentation**
*   **[ ] Phase 4: Evaluation & Interpretation**
*   **[ ] Phase 5: Documentation & Presentation**

---
*(This README will be updated as the project progresses.)*