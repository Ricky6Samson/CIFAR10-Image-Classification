🧠 CIFAR-10 Image Classification Project

This project walks through a complete end-to-end machine learning workflow, built around the CIFAR-10 image dataset.

It covers everything from data preparation and model training to experiment logging in SQL and interactive visualization in Power BI.
My goal here was to show how I approach ML projects not just from a modeling perspective, but from a real-world, end-to-end analytical pipeline point of view.

📂 Project Structure
CIFAR10-Project
│
├── Sample Data
│   ├── Class_Metrics
│   ├── Confusion_Matrix
│   ├── Epoch_Stats
│   ├── Model_Info
│   └── Predictions_Log
│
├── Notebooks
│   ├── Libraries.ipynb                      # All library imports
│   ├── Data Preparation.ipynb        # Data loading, normalization, augmentation
│   ├── Training.ipynb                     # Custom CNN + Transfer Learning (MobileNetV2)
│   ├── Metrics.ipynb                      # Evaluation and visualization
│   ├── SQL_Logging.ipynb            # Logging results into SQL
│
├── SQL
│   └── create_tables.sql             # Script to create the 5 SQL tables
│
├── CIFAR10-report.pbix               # Power BI dashboard
├── Dashboard.png                     # Dashboard preview
└── README.md

🚀 Overview

The CIFAR-10 dataset contains 60,000 color images (32x32 pixels) across 10 classes, including animals, vehicles, and objects.

In this project, I:

Trained two models — a Custom CNN and a Transfer Learning model using MobileNetV2.

Used data augmentation to improve generalization.

Logged all experiment details and metrics into a SQL database.

Built a Power BI dashboard to visualize training trends and compare models.

It’s meant to demonstrate both machine learning and data engineering thinking in one cohesive pipeline.

🧩 SQL Database Design

To make the experiment tracking structured and reusable, I designed five SQL tables:

Table	Description
Model_Info	Stores model names, training date, and accuracy scores.
Epoch_Stats	Logs epoch-level metrics like loss and accuracy.
Class_Metrics	Stores per-class precision, recall, and F1 scores.
Confusion_Matrix	Records actual vs. predicted values for misclassification analysis.
Predictions_Log	Keeps a detailed record of each prediction and whether it was correct.

This approach mirrors how real ML projects maintain experiment history for reproducibility and version tracking.

📊 Power BI Dashboard

The Power BI dashboard brings the SQL data to life with interactive visuals.
It includes:

KPI Cards for best model accuracy and training summary.

Trend charts for loss and accuracy over epochs.

Class-level metrics for detailed performance comparison.

Confusion matrix heatmap to spot where models struggle most.

Model comparison visuals showing overfitting and generalization gaps.

Dashboard Preview:


🧠 Model Development
1. Data Preparation & Augmentation

Before training, I normalized the pixel values and applied data augmentation to help models generalize better.
The augmentations included:

Random flips

Small rotations

Zooms and shifts

This simple step reduced overfitting and improved validation accuracy by around 2–3%.

2. Model Architectures

Custom CNN

3 convolutional blocks with BatchNorm and Dropout

Adam optimizer with learning rate decay

Early stopping to prevent overfitting

Transfer Learning (MobileNetV2)

Pretrained on ImageNet and fine-tuned on CIFAR-10

Froze the base layers initially, then unfroze top layers for fine-tuning

Used callbacks like EarlyStopping and ReduceLROnPlateau

The transfer learning model clearly outperformed the custom CNN, with better validation accuracy and less overfitting.

3. Optimization & Tracking

I used the following techniques to make training smoother and more reliable:

EarlyStopping to stop training when validation stopped improving

Learning rate scheduling to fine-tune model convergence

SQL-based logging to store results from each experiment

Power BI dashboards for at-a-glance model monitoring

🧾 Results Summary
Metric	Custom CNN	Transfer Learning
Train Accuracy	92%	95%
Validation Accuracy	83%	90%
Overfitting Gap	9%	1%
Best Epoch	24	17

Key Takeaways:

The transfer learning model generalized far better, even with fewer training epochs.

Data augmentation helped stabilize accuracy trends.

Most misclassifications happened between visually similar classes (e.g., dog vs. cat, truck vs. automobile).

🧰 Tech Stack
Category	Tools Used
Programming	Python, Jupyter Notebook
Libraries	TensorFlow, Keras, scikit-learn, NumPy, Pandas
Database	SQL Server
Visualization	Power BI
Dataset	CIFAR-10
💼 Why This Project Matters

This project is a reflection of how I approach machine learning practically:

It’s structured, not just experimental.

Results are tracked and explainable.

The final output (Power BI dashboard) translates metrics into business-ready visuals.

It shows the blend of data analysis, machine learning, and presentation that’s essential for real-world data roles.

👤 Author

Ricky Samson
Data & Machine Learning Enthusiast | Building real-world analytics pipelines
🔗 [LinkedIn: www.linkedin.com/in/ricky-samson-aa6569331
]