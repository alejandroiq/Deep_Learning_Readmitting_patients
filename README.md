## 📘 Deep Learning with Neural Networks - Hospital Patient Readmission

### 🧠 Project Overview

This project explores how to design, train, and evaluate a deep learning model to predict patient readmission using electronic health record (EHR) data. 
We implement a baseline neural network, apply multiple model improvements, and focus on performance metrics relevant to real-world healthcare applications.

---

### 🚑 Objective

To build a neural network that predicts whether a diabetic patient will be readmitted to the hospital, using structured data from the `readmission_data.csv` dataset. 
The goal is to maximize **recall**, reducing the chance of missing high-risk patients.

---

### 🔧 Techniques Implemented

#### 📌 Baseline Model

* Input shape based on tabular feature count
* Two hidden layers (`64 → 32`) using ReLU
* Sigmoid output for binary classification
* Loss: `binary_crossentropy`; Metric: `recall`

#### 📈 Training Monitoring

* Custom plot functions for visualizing loss and recall
* TensorBoard integration for real-time metrics and histograms

#### 🧪 Model Improvements

* **EarlyStopping**: Halts training when validation loss stops improving
* **ModelCheckpoint**: Saves best model weights based on val\_loss
* **ReduceLROnPlateau**: Dynamically reduces learning rate
* **TensorBoard**: Visual tool for inspecting training curves and layer activity
* **Dropout (0.2)**: Reduces overfitting
* **BatchNormalization**: Stabilizes and accelerates training
* **LeakyReLU**: Prevents dead neurons
* **Gradient Clipping**: Prevents unstable gradient explosions

---

### ⚕️ Healthcare Context

* **Recall** prioritized to capture as many true readmissions as possible
* Avoiding false negatives is critical for patient safety
* Techniques were chosen not only for performance but for **stability and reliability**, which are essential in healthcare AI

---

### 📊 Final Results

| Model                   | Test Recall |
| ----------------------- | ----------- |
| Baseline Model          | \~0.00      |
| Final Improved Model    | \~0.41      |
| Best Checkpointed Model | \~0.43      |

> EarlyStopping triggered around epoch 16, suggesting good convergence. The best-performing model significantly outperformed the baseline in identifying readmitted patients.

---

### 💡 Key Learnings

* Overfitting can be mitigated with the right combination of callbacks and regularization techniques.
* In healthcare, **data quality and relevance often matter more than model complexity**.
* Monitoring tools like TensorBoard are essential for diagnosing model behavior early in training.


  By Alejandro Silva
  
