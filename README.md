# Wafer Defect Detection using Deep Learning

This project focuses on detecting potential scratch defects on silicon wafer dies using deep learning techniques. It simulates a real-world semiconductor manufacturing scenario where identifying faulty dies early can improve production quality and reduce costs.

## 📌 Problem Statement

The goal is to classify each die in a wafer map as either **scratch-defective (1)** or **non-defective (0)** based on features derived from its position and neighboring dies.

---

## 📂 Project Structure

```bash
.
├── scratch_detection_roy_boker.ipynb         # Main Jupyter Notebook for training
├── data/
│   ├── df_wafers.csv         # Input wafer data
│   └── df_wafers_test.csv    # Unlabeled test data
├── outputs/
│   └── example_submission.csv
└── README.md
```

---

## 🧠 Model Workflow

### 1. Feature Engineering

* Used KDTree to compute for each die:

  * Number of bad neighbors within radius 5 and 9
  * Percentage of bad neighbors
  * Total number of neighbors
* Encoded die coordinates using `sin` and `cos` for spatial positioning

### 2. Model

* A fully connected feedforward neural network (MLP)
* Two hidden layers with dropout
* `sigmoid` activation for binary classification
* Loss: Binary Cross-Entropy with class weights

### 3. Evaluation

* Trained on \~5 million dies
* Achieved \~93% accuracy
* Detailed classification report with `precision`, `recall`, and `f1-score` per class

### 4. Prediction Output

* The final predictions were written to a CSV file that includes all original columns **plus** a new `IsScratchDie` prediction column.

---

## 🛠️ Tech Stack

* Python, NumPy, Pandas
* Scikit-learn (KDTree, preprocessing)
* TensorFlow / Keras (deep learning)
* Matplotlib (visualizations)

---

## 📁 Example Prediction Output

```
WaferName,DieX,DieY,IsGoodDie,...,IsScratchDie
7vbLWL,14,27,True,...,0
Ha3TvN,21,22,False,...,1
```

---

## 📈 Visual Results

* Plotted training vs validation accuracy and loss
* Analyzed confusion matrix for class-wise performance

---

## 🙋 Author

**Roy Boker**
Email: [royboker15@gmail.com](mailto:royboker15@gmail.com)
Date: May 2025

---

## 🚀 How to Run

1. Train the model via `train_model.ipynb`
2. Run predictions using the trained model on `df_wafers_test.csv`
