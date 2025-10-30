# CIFAR-10 Image Classification Project

This project demonstrates a complete **end-to-end machine learning workflow**, from data preparation and model training to SQL-based logging and Power BI visualization, using the **CIFAR-10** dataset.
It’s designed to showcase both **data engineering** and **machine learning** skills in a professional portfolio context.

---

##  Project Structure

```
CIFAR10-Project
│
├──  Sample Data
│   ├── Class_Metrics
│   ├── Confusion_Matrix
│   ├── Epoch_Stats
│   ├── Model_Info
│   └── Predictions_Log
│
├──  Notebooks
│   ├── Data Preparation.ipynb        # Functions for preparing CIFAR-10 data for CNN and Transfer Learning models
│   ├── Training.ipynb                # Custom CNN definition and training + transfer model training
│   ├── Metrics.ipynb                 # Prediction, classification report, and confusion matrix functions
│   ├── SQL_Logging.ipynb             # Snippets to log data into the 5 SQL tables
│   └── Libraries.ipynb               # All necessary library imports
│
├──  SQL
│   └── create_tables.sql             # Queries to create 5 database tables
│
├── CIFAR10-report.pbix               # Power BI report visualizing all 5 SQL tables
├── Dashboard.png                     # Screenshot of the final Power BI dashboard
└── README.md
```

---

##  Project Overview

The **CIFAR-10** dataset contains 60,000 images across 10 object classes.
This project involves:

1. Building and training two models — a **custom CNN** and a **transfer learning model** (using MobileNet_v2).
2. Logging model statistics and metrics into **SQL tables** for structured tracking.
3. Creating **interactive Power BI dashboards** to visualize model performance, accuracy trends, and class-wise results.

---

##  SQL Database Design

Five tables were used to store experiment data:

| Table                | Description                                                                            |
| -------------------- | -------------------------------------------------------------------------------------- |
| **Model_Info**       | General details about each trained model (date, name, training & validation accuracy). |
| **Epoch_Stats**      | Epoch-wise metrics such as loss and accuracy for each model.                           |
| **Class_Metrics**    | Precision, recall, F1-score, and support for each class of the best-performing model.  |
| **Confusion_Matrix** | True vs predicted class counts to evaluate misclassifications.                         |
| **Predictions_Log**  | Each image’s prediction result, including whether it was correct.                      |

---

##  Power BI Dashboard

The **CIFAR10-report.pbix** file visualizes key insights:

* **KPI Cards**: Best model accuracy, training vs validation performance.
* **Trend Charts**: Accuracy and loss over epochs.
* **Class Metrics**: Precision/Recall/F1-score comparisons.
* **Confusion Matrix Heatmap**: Misclassification analysis.
* **Model Comparison**: Train vs validation accuracy and overfitting gap.

A preview is included below:

![Dashboard Preview](Dashboard.PNG)

---

##  How to Use

1. **Clone or download** this repository.
2. **Create SQL tables** using `SQL/create_tables.sql`.
3. **Run the notebooks** in the `Notebooks/` folder sequentially:

   * `Libraries.ipynb`
   * `Data Preparation.ipynb`
   * `Training.ipynb`
   * `Metrics.ipynb`
   * `SQL_Logging.ipynb`
4. **Insert data** using the provided notebooks.
5. **Open Power BI** and load data from the SQL tables to explore the visuals.

##  Results Summary

* **Average Accuracy (Best Model):** ~90%
* **Average Overfitting Gap:** 1%
* **Key Insights:**

  * Transfer learning model achieved higher validation accuracy.
  * Certain classes (e.g., Dog, Cat) showed more confusion, as seen in the matrix.


---

##  Tech Stack

| Category          | Tools Used                                      |
| ----------------- | ----------------------------------------------- |
| **Programming**   | Python, Jupyter Notebook                        |
| **Libraries**     | TensorFlow / Keras, NumPy, Pandas, scikit-learn |
| **Database**      | SQL Server Management Studio                              |
| **Visualization** | Power BI                                        |
| **Dataset**       | CIFAR-10                                        |

---

##  Author

**Ricky Samson**
A data & ML enthusiast building end-to-end analytical solutions.
 [www.linkedin.com/in/ricky-samson-aa6569331]

---
