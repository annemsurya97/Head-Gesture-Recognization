from tkinter import *
import tkinter
import tkinter as tk
from tkinter import ttk
from tkinter import Text, Scrollbar
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
import os, joblib

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, model_from_json, Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.utils import to_categorical

global filename
global X_train, X_test, y_train, y_test
global X, Y
global le
global dataset
global classifier
global cnn_model, labels

def uploadDataset():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head())+"\n\n")

def preprocessDataset():
    global le
    global dataset, labels
    global X_train, X_test, y_train, y_test
    le = LabelEncoder()
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    print(dataset.info())
    text.insert(END,str(dataset.head())+"\n\n")
    
    dataset.head()
    labels = dataset['Miscare'].unique()
    print('labels:',labels)
    dataset['Miscare'] = le.fit_transform(dataset['Miscare'])
    print(dataset)
    
    #n_samples = 50000  # You can adjust this based on your need
    #dataset = resample(dataset, 
    #                        replace=True,    # Sample with replacement
    #                        n_samples=n_samples,  # Number of samples in the resampled set
    #                        random_state=42)
                            
    x = dataset.drop(['Miscare'], axis=1)
    y = dataset['Miscare']
    
    smote = SMOTE(random_state=30)
    
    X,Y = smote.fit_resample(x, y)

    text.insert(END,"Total records found in dataset: "+str(X.shape)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=44)
    text.insert(END,"Total records found in dataset to train: "+str(X_train.shape)+"\n\n")
    text.insert(END,"Total records found in dataset to test: "+str(X_test.shape)+"\n\n")
    text.insert(END, "Total records in dataset to X_test: " + str(X_test) + "\n")
    sns.set(style="darkgrid")  # Set the style of the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    ax = sns.countplot(x=Y, data=dataset, palette="Set3")
    plt.title("Count Plot") 
    plt.xlabel("Miscare Categories") 
    plt.ylabel("Count")  
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

    plt.show()
    
    
precision = []
recall = []
fscore = []
accuracy = []

#function to calculate various metrics such as accuracy, precision etc
def PerformanceMetrics(algorithm, testY,predict):
    global labels
    
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    print(algorithm+' Accuracy    : '+str(a))
    print(algorithm+' Precision   : '+str(p))
    print(algorithm+' Recall      : '+str(r))
    print(algorithm+' F1-SCORE      : '+str(f))
    text.insert(END, "Performance Metrics of " + str(algorithm) + "\n")
    text.insert(END, "Accuracy: " + str(a) + "\n")
    text.insert(END, "Precision: " + str(p) + "\n")
    text.insert(END, "Recall: " + str(r) + "\n")
    text.insert(END, "F1-SCORE: " + str(f) + "\n\n")
    report=classification_report(predict, testY,target_names=labels)
    print('\n',algorithm+" classification report\n",report)
    text.insert(END, "classification report: \n" + str(report) + "\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict,pos_label=1)
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    plt.title(algorithm+" ROC Graph")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def Perceptron_Model():
    global X_train, X_test, y_train, y_test
    global predict, model_folder
    
    Model_file = 'model/Perceptron_Model.pkl'
    
    if os.path.exists(Model_file):
        perceptron_classifier = joblib.load(Model_file)
        predict = perceptron_classifier.predict(X_test)
        PerformanceMetrics("Perceptron", y_test, predict)
    else:
        # Perceptron model with default parameters
        perceptron_classifier = Perceptron(
            penalty=None,             
            max_iter=1000,            
            tol=1e-3,                 
            random_state=42           
        )
        
        perceptron_classifier.fit(X_train, y_train)
        joblib.dump(perceptron_classifier, Model_file)
        predict = perceptron_classifier.predict(X_test)
        print("Perceptron model trained and model weights saved.")
        PerformanceMetrics("Perceptron", y_test, predict)

def MLP_Model():
    global X_train, X_test, y_train, y_test
    global predict, model_folder
    
    Model_file = 'model/MLP_Model.pkl'
    
    if os.path.exists(Model_file):
        mlp_classifier = joblib.load(Model_file)
        predict = mlp_classifier.predict(X_test)
        PerformanceMetrics("MLP Classifier", y_test, predict)
    else:
        # Perceptron model with default parameters
        mlp_classifier = MLPClassifier()

        mlp_classifier.fit(X_train, y_train)
        joblib.dump(mlp_classifier, Model_file)
        predict = mlp_classifier.predict(X_test)
        print("MLP model trained and model weights saved.")
        PerformanceMetrics("MLP Classifier", y_test, predict)

def DNN():
    global X_train,X_test,y_train,y_test, model
    model_path = 'model/dnn_model.h5'  # Define the model file path

    # Check if the model file exists
    if os.path.exists(model_path):
        print("Loading existing DNN model...")
        model = load_model(model_path)  # Load the pre-trained DNN model
    else:
        print("Training a new DNN model...")
        # Define the DNN model structure
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='softmax'))  # Assuming classification with 5 classes

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=40, verbose=0)

        # Save the trained model for future use
        model.save(model_path)  # Save the model in the specified path
        print(f"Model saved to {model_path}")

    # Extract features from the model (excluding the last layer)
    extractor = Sequential(model.layers[:-1])
    X_train_extracted = extractor.predict(X_train)
    X_test_extracted = extractor.predict(X_test)

    # Train a Decision Tree Regressor using the extracted features
    DTC = RandomForestClassifier()
    DTC.fit(X_train_extracted, y_train)
    y_pred = DTC.predict(X_test_extracted)

    # Insert results into the text widget (assuming you are using Tkinter for displaying the output)
    text.insert(tkinter.END, '\n\n---------DNN Model---------\n\n')   
    PerformanceMetrics("DNN Model", y_pred, y_test)

def predict():
    global model, labels
    path = filedialog.askopenfilename(initialdir="Datasets")
    
    # Load the test dataset
    test = pd.read_csv(path)
    text.insert(tkinter.END, '\n\n-------Test Dataset-------\n\n')
    text.insert(tkinter.END, '\n\n' + str(test) + '\n\n')
    
    if 'Miscare' in test.columns:
        test = test.drop(['Miscare'], axis=1)
    
    predictions = model.predict(test)
    
    # Convert predictions to class labels (if needed)
    predicted_classes = np.argmax(predictions, axis=1)  # For classification tasks with softmax output
    
    # Add the predictions to the DataFrame
    test['Prediction Activity'] = [labels[p] for p in predicted_classes] 
    
    # Display the updated test DataFrame with predictions
    text.insert(tkinter.END, '\n\n-------Prediction Results-------\n\n')
    text.insert(tkinter.END, '\n\n' + str(test) + '\n\n')

def graph():
    columns = ["Algorithm Name", "Accuracy", "Precision", "Recall", "f1-score"]
    algorithm_names = ["Perceptron Classifier", "MLP Classifier", "DenseNet+RF Model"]
    
    # Combine metrics into a DataFrame
    values = []
    for i in range(len(algorithm_names)):
        values.append([algorithm_names[i], accuracy[i], precision[i], recall[i], fscore[i]])
    
    temp = pd.DataFrame(values, columns=columns)
    text.delete('1.0', END)
    # Insert the DataFrame in the text console
    text.insert(END, "All Model Performance metrics:\n")
    text.insert(END, str(temp) + "\n")
    
    # Plotting the performance metrics
    metrics = ["Accuracy", "Precision", "Recall", "f1-score"]
    index = np.arange(len(algorithm_names))  # Positions of the bars

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2  # Width of the bars
    opacity = 0.8

    # Plotting each metric with an offset
    plt.bar(index, accuracy, bar_width, alpha=opacity, color='b', label='Accuracy')
    plt.bar(index + bar_width, precision, bar_width, alpha=opacity, color='g', label='Precision')
    plt.bar(index + 2 * bar_width, recall, bar_width, alpha=opacity, color='r', label='Recall')
    plt.bar(index + 3 * bar_width, fscore, bar_width, alpha=opacity, color='y', label='f1-score')

    # Labeling the chart
    plt.xlabel('Algorithm')
    plt.ylabel('Scores')
    plt.title('Performance Comparison of All Models')
    plt.xticks(index + bar_width, algorithm_names)  # Setting the labels for x-axis (algorithms)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
    
def close():
    main.destroy()

# Create the main application window
main = tk.Tk()
main.title("Head Gesture Recognition Using Inertial Sensors")
main.geometry("1100x700")

# Set the background color
main.config(bg='#1e272e')

# Title Label
title_font = ('Times New Roman', 16, 'bold')
title = tk.Label(
    main,
    text="Design and Implementation of Head Gesture Recognition System\nUsing 6DOF Inertial Sensors for Enhanced Interaction Control",
    bg='#ff793f',
    fg='white',
    font=title_font,
    pady=10,
)
title.pack(fill="x", padx=20, pady=10)

# Define button styling
button_font = ('Times New Roman', 12, 'bold')
button_style = {
    "bg": "#0fbcf9",
    "fg": "white",
    "activebackground": "#34ace0",
    "activeforeground": "white",
    "font": button_font,
    "relief": "raised",
    "width": 20,
}

# Button Grid Layout
button_frame = tk.Frame(main, bg='#1e272e')
button_frame.pack(pady=20)

# Row 1 Buttons
btn1 = tk.Button(button_frame, text="Upload Dataset", command=uploadDataset, **button_style)
btn1.grid(row=0, column=0, padx=15, pady=10)

btn2 = tk.Button(button_frame, text="Preprocess Dataset", command=preprocessDataset, **button_style)
btn2.grid(row=0, column=1, padx=15, pady=10)

btn3 = tk.Button(button_frame, text="Perceptron Classifier", command=Perceptron_Model, **button_style)
btn3.grid(row=0, column=2, padx=15, pady=10)

# Row 2 Buttons
btn4 = tk.Button(button_frame, text="MLP Classifier", command=MLP_Model, **button_style)
btn4.grid(row=1, column=0, padx=15, pady=10)

btn5 = tk.Button(button_frame, text="DenseNet with RF Classifier", command=DNN, **button_style)
btn5.grid(row=1, column=1, padx=15, pady=10)

btn6 = tk.Button(button_frame, text="Comparison Graph", command=graph, **button_style)
btn6.grid(row=1, column=2, padx=15, pady=10)

# Row 3 Buttons
btn7 = tk.Button(button_frame, text="Prediction", command=predict, **button_style)
btn7.grid(row=2, column=0, padx=15, pady=10)

btn8 = tk.Button(button_frame, text="Exit", command=close, **button_style)
btn8.grid(row=2, column=1, padx=15, pady=10)

# Text Box with Scrollbar
text_frame = tk.Frame(main, bg='#d2dae2', bd=2, relief='groove')
text_frame.pack(pady=20, padx=20, fill='both', expand=True)

scroll = Scrollbar(text_frame, orient="vertical")
text = Text(text_frame, wrap="word", yscrollcommand=scroll.set, font=("Times New Roman", 12), bg="white", fg="black")
scroll.pack(side="right", fill="y")
text.pack(side="left", fill="both", expand=True)
scroll.config(command=text.yview)

# Final Configuration
main.mainloop()