# Anomaly Detection in Medical Datasets: A Med-GEMMA Inspired Approach
Project for the Master in Biomedical Engineering (BME) Paris

This repository contains the code, research, and documentation for my Master's project focused on applying Machine Learning techniques to detect unusual patterns in medical datasets. The primary goal is to enhance clinical insights, potentially leading to early risk identification and optimized healthcare processes. This work is conceptually inspired by the principles of advanced healthcare AI models like Google's Med-GEMMA.

🎯 **Project Goal**
To develop, evaluate, and interpret machine learning models capable of identifying statistically significant anomalies in structured medical data (e.g., doctor appointments, clinical analyses).

**Key Objectives**
*   **Research & Understand:** Conduct a thorough literature review of anomaly detection techniques in healthcare.
*   **Preprocess Data:** Acquire, clean, and engineer features from public, anonymized medical datasets (e.g., MIMIC-IV, PhysioNet).
*   **Model Development:** Implement and train several ML models, from classical baselines (Isolation Forest) to more advanced approaches (Autoencoders, potentially leveraging transformer embeddings).
*   **Evaluate & Interpret:** Assess model performance using appropriate metrics and interpret the detected anomalies to determine their potential clinical relevance.
*   **Document:** Maintain a reproducible and well-documented workflow.

🚀 **Your First Steps: Where to Begin Right Now**
Here is a clear action plan to get you started immediately.

**1. Set Up Your Environment (1-2 hours):**
Follow the instructions in the "Getting Started" section below to clone the repository, create a virtual environment, and install the required packages. A clean environment prevents future headaches.

**2. Start Your Literature Review (3-5 hours):**
Begin reading about anomaly detection in healthcare. Don't just browse—take notes.
*   **Action:** Create a new file in the root directory named `LITERATURE_REVIEW.md`. Use it to summarize key papers, techniques, and datasets you find.
*   **Focus on:**
    *   What types of anomalies are common in clinical data (e.g., data entry errors, rare disease symptoms, fraudulent claims)?
    *   What are the pros and cons of common algorithms (Isolation Forest, Local Outlier Factor, One-Class SVM, Autoencoders)?
    *   How have others used datasets like MIMIC-IV or PhysioNet?

**3. Identify and Download Your First Dataset (2-3 hours):**
You can't do anything without data. This is a priority.
*   **Action:** Browse public medical datasets. PhysioNet is an excellent resource. The MIMIC-IV dataset is powerful but requires getting access credentials, which can take time. Start with something smaller and more accessible if needed.
*   **Action:** Once you've chosen a dataset, download its files and place them in the `data/raw/` directory.

**4. Begin Exploratory Data Analysis (EDA) (4-6 hours):**
Once the data is downloaded, it's time to get your hands dirty.
*   **Action:** Open the `notebooks/01_data_exploration.ipynb` notebook.
*   **Tasks:**
    *   Load the dataset using Pandas.
    *   Use `.head()`, `.info()`, and `.describe()` to get a first look.
    *   Check for missing values (`.isnull().sum()`).
    *   Create simple visualizations (histograms, box plots) to understand the distribution of key features.
    *   Document every finding and question you have directly in the notebook. This is your research diary.

📈 **Detailed Project Roadmap**
This roadmap breaks the project into five distinct phases. Follow it step-by-step.

**Phase 1: Foundation & Literature Review (Week 1-2)**
*   [x] Task 1.1: Set up the complete project environment (Git, venv, packages).
*   [x] Task 1.2: Conduct a comprehensive literature review. (Use `LITERATURE_REVIEW.md`).
*   [x] Task 1.3: Clearly define the type of "anomaly" to be detected. Is it a rare event, a data error, or an unexpected patient trajectory?
*   [x] Task 1.4: Select and gain access to one or two primary datasets (e.g., PhysioNet, MIMIC-IV). Place them in `data/raw/`.

**Phase 2: Data Acquisition & Preprocessing (Week 3-4)**
*   [x] Task 2.1: Perform detailed Exploratory Data Analysis (EDA) in `notebooks/01_data_exploration.ipynb`.
*   [x] Task 2.2: Develop data cleaning and preprocessing scripts.
    *   [x] Handle missing values (imputation or removal).
    *   [x] Correct data types (e.g., convert dates, ensure numerical columns are numeric).
    *   [x] Normalize or scale numerical features.
    *   [x] Encode categorical features (one-hot, label encoding).
*   [x] Task 2.3: Implement the preprocessing logic in `src/features/build_features.py`. (Fulfilled by notebook `02`)
*   [x] Task 2.4: Save the final, processed dataset to the `data/processed/` directory.

**Phase 3: Model Development & Experimentation (Week 5-7)**
*   [x] Task 3.1: Baseline Models
    *   [x] Implement and train an Isolation Forest model in `notebooks/03_model_development_baseline.ipynb`.
    *   [ ] Experiment with other baselines like Local Outlier Factor (LOF) or One-Class SVM.
    *   [ ] Save trained baseline models.
*   [x] Task 3.2: Advanced Models (Autoencoder)
    *   [x] Design and build a simple Autoencoder using TensorFlow or PyTorch in `notebooks/04_model_development_advanced.ipynb`.
    *   [x] Train the Autoencoder on the processed data. The reconstruction error will be your anomaly score.
    *   [ ] Tune hyperparameters (network architecture, learning rate, epochs).
*   [ ] Task 3.3: Formalize the training process in `src/models/train_model.py`.

**Phase 4: Evaluation & Interpretation (Week 8-9)**
*   [ ] Task 4.1: Define evaluation metrics. If you have some labeled anomalies, use Precision, Recall, and F1-score. If not, you'll rely on visual inspection and quantitative measures of separation.
*   [ ] Task 4.2: Evaluate the baseline and advanced models on a test set.
*   [ ] Task 4.3: For the top-performing model, investigate the highest-scored anomalies.
    *   What features contributed most to their anomaly scores?
    *   Do these anomalies make clinical sense? (This is the most important part of the project).
*   [ ] Task 4.4: Create visualizations (e.g., PCA/t-SNE plots) to show how anomalies are separated from normal data. Implement these in `src/visualization/visualize.py`.

**Phase 5: Documentation & Presentation (Week 10-12)**
*   [ ] Task 5.1: Write the final project report/manuscript (`reports/manuscript.pdf`).
    *   Introduction (motivation, problem statement).
    *   Methods (data, preprocessing, models).
    *   Results (model performance, key findings).
    *   Discussion (interpretation of anomalies, limitations, future work).
    *   Conclusion.
*   [ ] Task 5.2: Generate all necessary figures and tables for the report and save them in `reports/figures/`.
*   [ ] Task 5.3: Create a presentation summarizing the project.
*   [ ] Task 5.4: Clean up the codebase, ensure all notebooks are runnable, and write the final version of this README.md.

### 🌟 Future Work: Evolving from Anomaly Detection to Diagnostic Assistance

The successful completion of this project will yield a robust system for identifying statistically significant anomalies within structured medical data. This is a critical first step in flagging potential issues for clinical review. However, the logical evolution of this work is to move from *anomaly detection* to *diagnostic reasoning*.

The vision for a future iteration of this project is to build a multimodal diagnostic assistant capable of ingesting diverse data types to suggest potential causes or associated diseases.

**Key Components of this Future Vision:**

*   **Multimodal Data Integration:** The system would be enhanced to process not just structured data, but also:
    *   **Unstructured Text:** Analyzing free-text from doctor-patient conversations, clinical notes, and medical reports.
    *   **Medical Imagery:** Incorporating image analysis from scans like X-rays, MRIs, or pathology slides.

*   **Advanced AI Models:** To handle this complexity, the aystem would leverage state-of-the-art, purpose-built medical foundation models. This is where a tool like **Google's Med-GEMMA** would become essential. Its ability to reason across both text and images would be the core engine for this advanced system.

*   **Clinical Reasoning Instead of Statistical Outliers:** Instead of simply flagging a data point as "unusual," the future system would aim to answer the question: "Given the patient's symptoms, clinical notes, and X-ray results, what are the likely differential diagnoses?"

**Challenges & Scope:**
Implementing such a system presents significant challenges, including the need for massive computational resources (high-end GPUs for training/inference), access to comprehensive and linked multimodal datasets (like MIMIC-IV), and navigating complex validation pathways. For these reasons, this vision is scoped as "future work" and serves as the long-term, inspirational goal that this foundational project is building towards.

🛠️ **Tech Stack (Planned)**
*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Data Handling: Pandas, NumPy
    *   ML / DL: Scikit-learn, TensorFlow/PyTorch
    *   Visualization: Matplotlib, Seaborn, Plotly
*   **NLP/Embeddings (Optional):** Hugging Face transformers
*   **Environment:** Jupyter Notebooks, VS Code, Git

📂 **Repository Structure**
BME/
│
├── 📄 .gitignore
├── 📄 README.md
│
├── 📁 data/
│ ├── 📁 raw/ # Original, immutable data.
│ ├── 📁 processed/ # Cleaned and preprocessed data.
│ └── 📄 .gitkeep # Placeholder to keep the directory in git.
│
├── 📁 notebooks/
│ ├── 📁 01_data_exploration.ipynb
│ ├── 📁 02_data_preprocessing.ipynb
│ ├── 📁 03_model_development_baseline.ipynb
│ └── 📁 04_model_development_advanced.ipynb
│
├── 📁 src/
│ ├── 📁 data/
│ │ └── 📄 make_dataset.py # Scripts to download/generate data.
│ ├── 📁 features/
│ │ └── 📄 build_features.py # Scripts to create features.
│ ├── 📁 models/
│ │ ├── 📄 train_model.py # Scripts to train models.
│ │ └── 📄 predict_model.py # Scripts for inference.
│ └── 📁 visualization/
│ └── 📄 visualize.py # Scripts for visualizations.
│
├── 📁 reports/
│ ├── 📁 figures/ # Figures and plots for the final report.
│ └── 📄 manuscript.pdf # Final thesis/report.
│
├── 📄 requirements.txt # Project dependencies.
└── 📄 setup.py # To make the project installable (optional but good practice).


🚀 **Getting Started**
**Prerequisites**
*   Python 3.8+
*   `pip` package manager

**Installation**
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/BME.git
    cd BME
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

(This README will be updated as the project progresses.)