# Parkinson's Disease Detection Using CNN & Transfer Learning

Early, non-invasive detection of Parkinson's Disease (PD) through deep learning analysis of handwritten spiral and meander drawings — benchmarking 5 CNN architectures with the best model achieving **93.43% accuracy**.

---

## Overview

Parkinson's Disease affects motor control, causing subtle changes in handwriting patterns that are detectable through image analysis. This project uses the **NewHandPD dataset** of handwritten drawings from 66 participants (35 healthy, 31 PD patients) to train and compare multiple CNN + Transfer Learning models for binary classification.

**Presented at VIT Industry Conclave 2024 | Published — VIT Conference, Chennai**

---

## Results

### Handwritten Drawing Classification (Spiral & Meander)

| Model | Accuracy (Spiral) | Accuracy (Meander) |
|---|---|---|
| Basic CNN | 75.76% | 75.76% |
| CNN + VGG16 | 75.76% | 72.73% |
| CNN + InceptionResNetV2 | 81.82% | 75.76% |
| CNN + InceptionV3 | **87.88%** | **83.33%** |

### Recurrence Plot Preprocessing (Spiral RP)

| Model | Accuracy |
|---|---|
| Basic CNN | 75.76% |
| MobileNetV2 | **93.43%** ✅ Best Result |

> **Key Finding:** Using Recurrence Plot transformation as a preprocessing step before MobileNetV2 improved accuracy from 75.76% to 93.43% — a 17+ percentage point gain over the baseline CNN.

---

## Dataset

- **Name:** NewHandPD Dataset (improved version of HandPD)
- **Source:** [UNESP — NewHandPD](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/)
- **Participants:** 66 people — 35 Healthy, 31 PD patients
- **Examinations per participant:** 12 total
  - 4 spiral drawings
  - 4 meander drawings
  - 2 circular motions (air + paper)
  - Diadochokinesis (left + right hand)

---

## Methodology

```
Handwritten Images (Spiral / Meander)
        ↓
Data Splitting (Train / Test)
        ↓
Image Augmentation (shear, zoom, horizontal flip)
        ↓
CNN + Transfer Learning Models
  ├── Basic CNN
  ├── VGG16
  ├── InceptionResNetV2
  ├── InceptionV3
  └── MobileNetV2 (with Recurrence Plot input)
        ↓
Binary Classification: Healthy vs PD
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Models | CNN, VGG16, InceptionV3, InceptionResNetV2, MobileNetV2 |
| Data Processing | NumPy, Pandas, ImageDataGenerator |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab, Jupyter Notebook |

---

## Project Structure

```
├── pd_cnn.ipynb                  # Main notebook — all models
├── data/
│   ├── Spiral splitted/
│   │   ├── train/
│   │   └── val/
│   └── Meander splitted/
├── results/
│   ├── confusion_matrices/
│   └── accuracy_loss_plots/
└── README.md
```

---

## Key Implementation Details

**Data Augmentation**
```python
ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
```

**Transfer Learning Setup**
- Pretrained weights: ImageNet
- Frozen base layers + custom classification head
- Custom head: GlobalAveragePooling2D → Dropout(0.5) → Dense(512) → Dense(1, sigmoid)

**Training Strategy**
- Optimizer: Adam with learning rate scheduling (halved every 5 epochs)
- Early stopping: patience=10, restore best weights
- Loss: Binary cross-entropy

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/akanksha-bhagat/Detection-of-Parkinson-s-Disease-Using-Machine-Learning.git
cd Detection-of-Parkinson-s-Disease-Using-Machine-Learning
```

2. Install dependencies
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn
```

3. Download the NewHandPD dataset from [UNESP](https://wwwp.fc.unesp.br/~papa/pub/datasets/Handpd/) and place it in the `data/` folder

4. Open and run `pd_cnn.ipynb` in Google Colab or Jupyter Notebook

---

## Evaluation

Each model was evaluated using:
- Test accuracy and loss
- Confusion matrix (Healthy vs PD predictions)
- Training and validation accuracy/loss curves

---

## Publication

This work was presented at the **VIT Industry Conclave 2024** and published at a VIT Conference, Chennai.

**Authors:** Akanksha Vikram Bhagat, Dr. Nayeemulla Khan
**Institution:** School of Computer Science & Engineering, Vellore Institute of Technology, Chennai

---

## Future Work

- Fine-tune pretrained layers instead of full freezing
- Expand dataset with more participants for better generalization
- Explore explainability tools (Grad-CAM) to visualize which drawing regions activate the model
- Deploy as a web-based screening tool using Flask or Streamlit
