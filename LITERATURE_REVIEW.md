


# Anomaly Detection in Healthcare – Key Papers

### VAE-IF: Deep feature extraction with averaging for fully unsupervised artifact detection in routinely acquired ICU time-series (Haule *et al.*)

**Summary:** Introduces **VAE-IF**, a hybrid model combining a variational autoencoder (VAE) and Isolation Forest (IF) to detect artifacts (anomalies) in minute-level ICU vital-sign time series without labeled data. The VAE learns compressed representations of physiological signals, and the IF uses these learned features to isolate anomalies. The model applies an averaging of latent features (“latent sample representations”) to enhance accuracy. Experiments on real ICU datasets show VAE-IF achieves performance comparable to fully supervised methods for artifact (noise) detection in heart rate, blood pressure, etc., demonstrating robust unsupervised anomaly detection.
**Relevance:** Directly targets unsupervised anomaly detection in critical care time-series (vital signs), aligning with clinical data applications (e.g. MIMIC waveform data). It exemplifies how combining deep learning (VAE) with classical unsupervised algorithms (Isolation Forest) can flag abnormal ICU signals without annotated anomalies.

* **Link:** [arXiv:2312.05959](https://arxiv.org/abs/2312.05959)
* **Citation:** Haule, H., Piper, I., Jones, P., Qin, C., Lo, T. Y. M., & Escudero, J. (2023). *VAE-IF: Deep feature extraction with averaging for fully unsupervised artifact detection in routinely acquired ICU time-series*. arXiv preprint arXiv:2312.05959.

### (Predictable) Performance Bias in Unsupervised Anomaly Detection (Meissen *et al.*)

**Summary:** Examines fairness in unsupervised anomaly detection for medical imaging. Using chest X-ray datasets (including MIMIC-CXR), the authors evaluate how UAD model performance varies across demographic subgroups. They observe *“fairness laws”*: linear relations between subgroup representation and anomaly-detection AUROC, and persistent performance gaps even with balanced data. They introduce a subgroup-AUROC metric to quantify bias. The study shows that simply balancing demographics is insufficient; some groups (e.g. certain age/race) remain harder to model, causing UAD models to underperform on those subpopulations.
**Relevance:** Highlights an important dimension of clinical anomaly detection: models must generalize across diverse patient subgroups. Although focused on imaging, it underscores the need to consider dataset composition and fairness when applying unsupervised detectors (e.g. autoencoders, isolation forests) to healthcare data. This informs the design and evaluation of anomaly detectors on MIMIC and PhysioNet datasets by reminding researchers to check for performance disparities.

* **Link:** [arXiv:2309.14198](https://arxiv.org/abs/2309.14198) (preprint of *EBioMedicine* 2024)
* **Citation:** Meissen, F., Breuer, S., Knolle, M., Buyx, A., Müller, R., Kaissis, G., & Rückert, D. (2024). *(Predictable) Performance Bias in Unsupervised Anomaly Detection*. *EBioMedicine*, 92, 105002.

### Harnessing EHRs for Diffusion-based Anomaly Detection on Chest X-rays (Kim *et al.*)

**Summary:** Proposes **Diff3M**, a multimodal anomaly detection framework that integrates chest X-ray images and structured Electronic Health Record (EHR) data via diffusion models. An *Image–EHR cross-attention* module conditions the reverse diffusion process on patient demographics and vitals, helping the model distinguish normal anatomical variation from true pathology. Additionally, a novel *Pixel-level Checkerboard Masking* strategy is used during reconstruction to better highlight anomalies. Evaluated on large chest radiograph collections (CheXpert and MIMIC-CXR IV), Diff3M achieves state-of-the-art unsupervised anomaly-detection performance. Notably, it demonstrates that fusing imaging with MIMIC-IV clinical data (e.g. BMI, blood pressure) improves detection over image-only models.
**Relevance:** Aligns exactly with clinical datasets (MIMIC-CXR/IV) and unsupervised methods. It shows a cutting-edge use of generative models for anomaly detection in healthcare, and highlights how combining data modalities (image + EHR) can boost anomaly-flagging. The techniques (diffusion model, attention) complement more classic autoencoder or tree-based methods, enriching the toolkit for clinical anomaly detection tasks.

* **Link:** [arXiv:2505.17311](https://arxiv.org/abs/2505.17311) (preprint, May 2025)
* **Citation:** Kim, H., Wang, Y., Ahn, M., Choi, H., Zhou, Y., & Hong, C. (2025). *Harnessing EHRs for Diffusion-based Anomaly Detection on Chest X-rays*. arXiv:2505.17311.

### Unsupervised Anomaly Detection in Medical Images with a Memory-augmented Multi-level Cross-attentional Masked Autoencoder (Tian *et al.*)

**Summary:** Introduces **MemMC-MAE**, a reconstruction-based anomaly detector for medical images. It’s a masked autoencoder (MAE) built with transformer layers augmented by a learnable memory bank and multi-level cross-attention. Large portions of each input image are randomly masked during training, forcing the model to reconstruct normal regions. Anomalous structures (e.g. tumors) that aren’t seen during training remain hard to reconstruct. Empirically, MemMC-MAE achieves *state-of-the-art* detection and localization on diverse medical imaging datasets (including colonoscopy images and chest X-rays for pneumonia and COVID-19). The memory mechanism helps the model “remember” normal patterns, so anomalies yield high reconstruction errors.
**Relevance:** Demonstrates powerful unsupervised anomaly detection in medical imaging, complementing traditional autoencoders. Its success on clinical image data suggests similar masked-AE ideas could be adapted to other medical time-series or waveforms (e.g. by masking segments of vital-sign signals). It serves as an inspiration for using deep reconstruction models when labels are scarce.

* **Link:** [arXiv:2203.11725](https://arxiv.org/abs/2203.11725) (MICCAI MLMI 2023)
* **Citation:** Tian, Y., Pang, G., Liu, Y., Wang, C., Chen, Y., Liu, F., Singh, R., Verjans, J. W., Wang, M., & Carneiro, G. (2023). *Unsupervised Anomaly Detection in Medical Images with a Memory-augmented Multi-level Cross-attentional Masked Autoencoder*. In *Medical Image Computing and Computer Assisted Intervention (MICCAI)* (pp. 1–12).

### Anatomy-aware Self-supervised Learning for Anomaly Detection in Chest Radiographs (Sato *et al.*)

**Summary:** Proposes **AnatPaste**, a self-supervised anomaly-augmentation method tailored to chest X-rays. It first segments lung fields and then pastes realistic “fake anomalies” within the lungs during pretraining, teaching the model to recognize subtle pathological shadows. This anatomy-guided augmentation yields more medically plausible anomalies than generic cut-and-paste techniques. The authors train a one-class classifier with AnatPaste pretraining and report high AUROC on three public chest X-ray datasets (∼92%, 79%, 82% respectively), outperforming other unsupervised detectors. This is the first work to leverage organ segmentation in pretext tasks for anomaly detection, showing that incorporating clinical anatomy significantly boosts detection accuracy.
**Relevance:** Focuses on chest radiographs (e.g. MIMIC-CXR) and uses unsupervised/self-supervised learning. It underscores the value of using domain-specific knowledge (lung anatomy) in unsupervised anomaly models. Researchers working on clinical anomaly detection can adapt the idea of anatomically informed augmentation or pretext tasks to other modalities (e.g. ECG intervals, ICU waveforms) to improve performance.

* **Link:** [iScience 26(7):107086 (2023)](https://www.cell.com/iscience/fulltext/S2589-0042%2823%2901163-X)
* **Citation:** Sato, J., Suzuki, Y., Wataya, T., Nishigaki, D., Kita, K., Yamagata, K., Tomiyama, N., & Kido, S. (2023). *Anatomy-aware self-supervised learning for anomaly detection in chest radiographs*. *iScience*, 26(7), 107086.

### Semantic Anomaly Detection in Medical Time Series (Festag & Spreckelsen)

**Summary:** Presents an unsupervised deep-learning approach to detect anomalies in medical time series (e.g. ECG). The method uses a denoising RNN autoencoder to embed short signal segments into a latent space, then applies clustering in that space. A novel “cluster-based similarity partitioning” ensemble identifies outlying cluster patterns. On real ECG data (from PhysioNet and MIMIC waveforms), the best system achieved an adjusted Rand index of ≈0.11, corresponding to about 72% precision/recall for anomaly detection. It outperformed several standard outlier methods, demonstrating that a latent-space clustering of RNN-encoded medical signals can isolate anomalous intervals.
**Relevance:** This work targets physiological waveforms (ECG) similar to those in PhysioNet/MIMIC. It exemplifies an unsupervised pipeline: autoencoding + clustering. The explicit handling of sequential medical data suggests this approach could be extended to other clinical time series (BP, SpO₂) for anomaly flagging. Its success shows that domain-agnostic deep encoding plus clustering can work on varied medical signals.

* **Link:** [IOS Press (Open Access, 2021)](https://ebooks.iospress.nl/doi/10.3233/SHTI210059)
* **Citation:** Festag, S., & Spreckelsen, C. (2021). *Semantic Anomaly Detection in Medical Time Series*. In **German Medical Data Sciences** (pp. 118–124). IOS Press.

### End-to-End Self-Tuning Self-Supervised Time Series Anomaly Detection (Deforce *et al.*)

**Summary:** Introduces **TSAP**, a self-supervised anomaly detection framework that automatically tunes its data-augmentation parameters without labels. TSAP treats anomaly detection as a binary task: augment “pseudo-anomalies” via parameterized transformations of normal time-series data, then train a detector to distinguish original vs. augmented series. Crucially, TSAP implements a differentiable augmentation model and an unsupervised validation loss (based on Wasserstein distance) to tune augmentation hyperparameters. In experiments (including PhysioNet ECG tasks), TSAP consistently outperforms fixed-augmentation baselines and other state-of-the-art SSL anomaly detectors across diverse anomaly types.
**Relevance:** While general to time series, TSAP includes experiments on physiological signals (PhysioNet 2017 ECG data) and targets “patient biomarker” series, aligning with clinical data use. It demonstrates the power of end-to-end self-supervision for unsupervised anomaly detection. Clinical anomaly detection systems could adapt TSAP’s idea to automatically select suitable synthetic anomaly models for ICU or EHR time series, improving robustness without manual tuning.

* **Link:** [arXiv:2404.02865](https://arxiv.org/abs/2404.02865) (SIAM J. Math Data Sci. 2025)
* **Citation:** Deforce, B., Lee, M.-C., Baesens, B., Serral Asensio, E., Yoo, J., & Akoglu, L. (2025). *End-To-End Self-Tuning Self-Supervised Time Series Anomaly Detection*. *SIAM Journal on Mathematics of Data Science*, arXiv:2404.02865.

### Unsupervised Anomaly Detection of Implausible Electronic Health Records: A Real-World Evaluation in Cancer Registries (Röchner & Rothlauf)

**Summary:** Applies unsupervised anomaly detection to identify implausible entries in structured EHR data (cancer registry records). The authors compare a pattern-based method (FindFPOF) and a neural autoencoder on 21,104 patient records (breast/colorectal/prostate cancer). Evaluated by clinical experts, both methods significantly enriched anomalous records compared to random. For example, each method flagged samples with \~28% implausible rates (precision 28%), versus 8% in a random sample. The autoencoder achieved \~22% sensitivity and 94% specificity. The study shows unsupervised models can reduce expert review effort by \~3.5× for data quality audits.
**Relevance:** Demonstrates unsupervised detection on high-dimensional clinical tabular data (EHR fields). It highlights that autoencoder-based methods can flag logical inconsistencies in patient records without labels. Though focused on data plausibility rather than physiological signals, it is highly relevant to anomaly detection in healthcare data. Its methodology (feature compression + outlier scoring) is applicable to any clinical dataset (e.g. EHR or lab values).

* **Link:** [BMC Med Res Methodol 23:125 (2023)](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-023-01946-0)
* **Citation:** Röchner, P., & Rothlauf, F. (2023). *Unsupervised anomaly detection of implausible electronic health records: a real-world evaluation in cancer registries*. *BMC Medical Research Methodology*, 23, 125.

### Explainable Unsupervised Anomaly Detection for Healthcare Insurance Data (De Meulemeester *et al.*)

**Summary:** Presents an unsupervised anomaly detection pipeline for health insurance billing data, aimed at identifying waste/fraud. The workflow uses **categorical embeddings** to vectorize provider records, applies a suite of unsupervised detectors (e.g. Isolation Forest, Local Outlier Factor), and then uses SHAP values for interpretable explanations of flagged anomalies. On real Belgian insurance data (11,851 general practitioners, 0.3% known outliers), embedding improved detection performance, and using SHAP allowed experts to trace which features (e.g. procedure codes) drove an anomaly score. The authors demonstrate state-of-the-art unsupervised methods (including ensemble techniques) outperform traditional ones, and SHAP greatly aids interpretability. For example, visualizing Shapley feature contributions helped experts confirm true outliers quickly.
**Relevance:** Although focused on billing data, this work is a practical example of unsupervised anomaly detection in a healthcare context. It shows how to handle categorical clinical data (using embeddings) and stresses the importance of explainability. Techniques like embedding categorical EHR fields and post-hoc explanation could be applied to any clinical anomaly task (e.g. in EHR or sensor data) to improve both detection quality and user trust.

* **Link:** [BMC Med Inform Decis Mak 24:40 (2024)](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02823-6)
* **Citation:** De Meulemeester, H., De Smet, F., Daemers, D., & Baesens, B. (2024). *Explainable unsupervised anomaly detection for healthcare insurance data*. *BMC Medical Informatics and Decision Making*, 24, 40.
