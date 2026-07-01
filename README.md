# Machine Learning & Deep Learning Intership Assignments
**Name:** Ashutosh Singh

**Internship:** MP Online AI & ML Internship

**Batch:** Batch 2

A collection of four end-to-end machine learning and deep learning projects covering image classification, face recognition, medical imaging, and tabular income prediction.

---

## 📂 Repository Structure

```
├── breast_cancer_detection_cnn.ipynb
├── Intel_Image_Classification_CNN.ipynb
├── Face_Recognition_CNN_LFW.ipynb
├── adult_census_income_ml.ipynb
└── README.md
```

---

## 📓 Assignments Overview

### 1. Breast Cancer Detection — CNN with Transfer Learning
**Notebook:** `breast_cancer_detection_cnn.ipynb`

Detects malignant vs. benign breast tumors from histopathological microscopy images using a fine-tuned **EfficientNetB0** backbone.

- **Dataset:** [BreakHis — Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset) (via KaggleHub)
- **Model:** EfficientNetB0 with a custom classification head; two-phase training — head-only first, then fine-tuning top layers
- **Input size:** 224 × 224 × 3
- **Key techniques:** Transfer learning, class-weight balancing, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Metrics:** Accuracy, F1-Score, Confusion Matrix, Classification Report

---

### 2. Intel Image Classification — Custom CNN
**Notebook:** `Intel_Image_Classification_CNN.ipynb`

Classifies natural scene images (buildings, forests, glaciers, mountains, seas, streets) into 6 categories using a custom-built CNN.

- **Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) (via KaggleHub)
- **Model:** Custom CNN with convolutional blocks, batch normalization, dropout, and L2 regularization
- **Key techniques:** ImageDataGenerator for augmentation, ROC-AUC per class (OvR), Grad-CAM visualizations
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC Curves, Confusion Matrix

---

### 3. Face Recognition — CNN on LFW Dataset
**Notebook:** `Face_Recognition_CNN_LFW.ipynb`

Identifies individuals from face images using a CNN trained on the Labeled Faces in the Wild (LFW) dataset.

- **Dataset:** [LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) (via KaggleHub)
- **Model:** Custom CNN with per-person classification head; classes filtered to individuals with ≥ 20 images
- **Input size:** 128 × 128 × 3
- **Data split:** 70% train / 15% validation / 15% test
- **Key techniques:** LabelEncoder serialized with pickle, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

### 4. Adult Census Income Prediction — Classical ML
**Notebook:** `adult_census_income_ml.ipynb`

Binary classification task to predict whether an individual earns more or less than $50K/year based on census demographic data.

- **Dataset:** [Adult Census Income (UCI)](https://www.kaggle.com/datasets/uciml/adult-census-income) (via KaggleHub) — 48,842 rows × 15 columns
- **Models compared:** Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), Support Vector Machine (SVM)
- **Key techniques:** Missing value handling, label encoding, standard scaling, SelectKBest feature selection, class imbalance analysis
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, ROC Curves

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Deep learning model building and training |
| scikit-learn | Classical ML models, preprocessing, metrics |
| OpenCV + Pillow | Image loading and preprocessing |
| NumPy / Pandas | Numerical computing and data manipulation |
| Matplotlib / Seaborn | Visualization and EDA plots |
| KaggleHub | Automated dataset downloading |

---

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.8+
- A Kaggle account (for KaggleHub dataset downloads)

### Install dependencies
```bash
pip install tensorflow scikit-learn opencv-python pillow numpy pandas matplotlib seaborn kagglehub
```

### Configure Kaggle credentials
```bash
# Place your kaggle.json API token at:
~/.kaggle/kaggle.json
```

### Run a notebook
Open any `.ipynb` file in Jupyter Notebook, JupyterLab, or Google Colab and run all cells from top to bottom. Datasets are downloaded automatically via KaggleHub on the first run.

---

## 📊 Key Results Summary

| Assignment | Task Type | Best Model / Backbone | Primary Metric |
|---|---|---|---|
| Breast Cancer Detection | Binary Classification | EfficientNetB0 (fine-tuned) | F1-Score |
| Intel Image Classification | 6-class Classification | Custom CNN | Accuracy / ROC-AUC |
| Face Recognition (LFW) | Multi-class Classification | Custom CNN | Accuracy / F1-Score |
| Adult Census Income | Binary Classification | Random Forest / SVM | ROC-AUC / F1-Score |

---

## 📝 Notes

- All notebooks use `SEED = 42` for full reproducibility.
- GPU acceleration is detected and configured automatically where available.
- Model checkpoints (best weights) are saved during training for each deep learning notebook.

## Author

Ashutosh Singh

Prepared as part of the MP Online AI & ML Internship.

