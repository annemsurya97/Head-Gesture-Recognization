*Head Gesture Recognition using 6DOF Inertial Sensors*

This project demonstrates how head gestures can be recognized from inertial sensor data (Roll, Pitch, Yaw, and Time) using machine learning and deep learning models.

We built an interactive Tkinter-based desktop app that lets you:
✔ Upload and preprocess sensor data
✔ Train and evaluate models (Perceptron, MLP, and DNN+RF hybrid)
✔ Visualize performance metrics and confusion matrices
✔ Compare models in a single graph
✔ Predict gestures on new test data

📂 Dataset Example

The dataset is a time-series of head movements captured as Roll, Pitch, and Yaw. Each row represents one reading, labeled with a gesture (Miscare).

Roll,Pitch,Yaw,Miscare,Time
4,-3,0,MoveRight_2s,0
4,-3,0,MoveRight_2s,0.02
5,-3,-1,MoveRight_2s,0.04
6,-2,-6,MoveRight_2s,0.34
8,-2,-18,MoveRight_2s,0.72


Roll, Pitch, Yaw → Head orientation (sensor readings)

Miscare → Gesture label (e.g., MoveRight_2s)

Time → Timestamp of reading

⚙️ Features

📁 Upload Dataset – load CSV files of sensor data

🧹 Preprocess Data – clean missing values, balance dataset with SMOTE, split into train/test

🤖 Train Models:

Perceptron Classifier

MLP Classifier

DNN with Random Forest Classifier (hybrid)

📊 Performance Metrics – accuracy, precision, recall, F1-score

📈 Comparison Graph – side-by-side comparison of models

🔮 Prediction Mode – test on new unseen data

🚀 How It Works

Data Upload & Cleaning

Load dataset in .csv format

Handle missing values and label encode gestures

Data Balancing

Use SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution

Training Models

Perceptron → baseline linear model

MLP → neural network for classification

DNN+RF → deep feature extractor + random forest classifier

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Visualizations: Confusion Matrix, ROC Curve, Performance Bar Graph

Prediction

Load new sensor data

Predict gesture labels

🛠️ Installation

Make sure you have Python 3.8+ installed.

# Install dependencies
pip install -r requirements.txt

📦 Requirements

numpy

pandas

seaborn

matplotlib

scikit-learn

imblearn

tensorflow

tkinter (comes pre-installed with Python on most systems)

▶️ Run the Application
python main.py


This will launch the Tkinter GUI:

Upload your dataset

Preprocess it

Train models

Compare results

Run predictions on new data

📊 Example Output

Confusion Matrix
<img width="956" height="751" alt="Perceptron" src="https://github.com/user-attachments/assets/72cf9257-1e98-4228-bb84-9ac4e0f24279" />
<img width="500" height="500" alt="MLP" src="https://github.com/user-attachments/assets/a6ee664d-c834-4f8b-a877-26e52ea7ae71" />
<img width="867" height="751" alt="DNN" src="https://github.com/user-attachments/assets/43e4b069-d071-4e7f-8e4e-1e4c5ccabd7d" />

Model Comparison Graph
<img width="1000" height="600" alt="Comaparision Graph" src="https://github.com/user-attachments/assets/57414838-23f4-4b54-aa2b-1a284eb7a1ff" />

