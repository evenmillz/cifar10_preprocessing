# 📊 Dataset Assignments Repository

This repository contains a collection of Python projects and assignments focused on data exploration, preprocessing, and machine learning workflows using datasets. These assignments demonstrate key concepts like data cleaning, forecasting, visualization, and evaluation using modern tools like **pandas**, **Prophet**, and **scikit-learn**.

---

## 📁 Repository Structure

```
dataset_assignments/
├── cifar10_project/
│   ├── cifar-10-batches-py/            # Raw CIFAR-10 binary batch files
│   ├── data/                           # Processed dataset outputs
│   ├── X_train.npy                     # Normalized feature data
│   ├── y_train.npy                     # Corresponding label data
│   ├── preprocess_cifar10.py          # Initial dataset loader & normalizer
│   ├── preprocess_cifar10_dataset.py  # Combined preprocessing script
│   └── verify_preprocessed_data.py    # Visualization & label mapping
│
├── tsp_tutorial_assignment/
│   ├── daily_temperature.csv          # Raw temperature dataset
│   └── temperature_prediction.py      # Prophet forecasting & visualization
│
└── README.md                          # You're here!
```

---

## ✅ Completed Assignments

### 1. **CIFAR-10 Dataset Preprocessing**

- Extracted and loaded CIFAR-10 image batches
- Combined and reshaped training data
- Normalized pixel values and saved to `.npy`
- Visualized 15 labeled images with category names

### 2. **Time Series Prediction (Prophet Tutorial)**

- Forecasted daily temperatures for New York using Facebook Prophet
- Plotted actual vs predicted values
- Calculated and printed Mean Absolute Error (MAE)

---

## 📦 Tools Used

- `pandas`
- `Prophet`
- `matplotlib`
- `sklearn`

---

## 📌 Notes

- Large data files (`.npy`, `.tar.gz`) are excluded from GitHub using `.gitignore`
- All code is written in Python 3.12 and executed in VS Code with Anaconda

---

## 🔄 Upcoming Sections

- Time Series Challenge
- YAML Exploration Assignment
- Custom Dataset Projects
