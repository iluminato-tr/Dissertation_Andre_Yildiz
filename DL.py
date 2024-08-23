# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:15:07 2024

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

# Example data: Replace with your actual DataFrame
encoded_data = pd.read_csv(r'~/Desktop/BRCA_project/results/encoded_data.csv', header=None)
encoded_data = encoded_data.astype(float)
multi = pd.read_csv(r"~/Desktop/BRCA_project/Processed/Multi_code.csv", index_col=0)
inter = pd.read_csv(r'~/Desktop/BRCA_project/results/intermediate.csv', header=None)
inter = inter.astype(float)
# Split the data into features and labels (change between intermediate and early data)
X = inter.values #encoded_data.values
y = multi['Multi_code'].values

# One-hot encode the labels
y = to_categorical(y)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, np.argmax(y_train, axis=1))

y_train_smote = to_categorical(y_train_smote)

model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_smote.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(y_train_smote.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Custom callback to store metrics
class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.auc_scores = []
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []

    def on_epoch_end(self, epoch, logs=None):
        y_val_pred = self.model.predict(X_val)
        y_val_pred_classes = np.argmax(y_val_pred, axis=1)
        y_val_true_classes = np.argmax(y_val, axis=1)

        auc = roc_auc_score(y_val, y_val_pred, multi_class='ovr')
        f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='weighted')
        precision = precision_score(y_val_true_classes, y_val_pred_classes, average='weighted')
        recall = recall_score(y_val_true_classes, y_val_pred_classes, average='weighted')
        
        self.epochs.append(epoch + 1)
        self.auc_scores.append(auc)
        self.f1_scores.append(f1)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)

metrics_callback = MetricsCallback()

history = model.fit(X_train_smote, y_train_smote,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    callbacks=[metrics_callback])

# Plotting the metrics
plt.figure(figsize=(10, 6))
plt.plot(metrics_callback.epochs, metrics_callback.auc_scores, label='AUC', color='blue')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('AUC Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('AUC_Over_Epochs.png')  # Save plot to file
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(metrics_callback.epochs, metrics_callback.f1_scores, label='F1 Score', color='green')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('F1_Score_Over_Epochs.png')  # Save plot to file
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(metrics_callback.epochs, metrics_callback.precision_scores, label='Precision', color='red')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('Precision_Over_Epochs.png')  # Save plot to file
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(metrics_callback.epochs, metrics_callback.recall_scores, label='Recall', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('Recall_Over_Epochs.png')  # Save plot to file
plt.show()

# Evaluate on test data
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# Compute and print metrics for test data
test_auc = roc_auc_score(y_test, y_test_pred, multi_class='ovr')
test_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
test_f1_macro = f1_score(y_test_true_classes, y_test_pred_classes, average='macro')
test_precision = precision_score(y_test_true_classes, y_test_pred_classes, average='weighted')
test_recall = recall_score(y_test_true_classes, y_test_pred_classes, average='weighted')

print(f'Test AUC: {test_auc:.4f}')
print(f'Test F1 Score (Weighted): {test_f1:.4f}')
print(f'Test F1 Score (Macro): {test_f1_macro:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')

# Confusion Matrix for Test Data
conf_matrix_test = confusion_matrix(y_test_true_classes, y_test_pred_classes)
disp_test = ConfusionMatrixDisplay(conf_matrix_test, display_labels=np.arange(y_test.shape[1]))

plt.figure(figsize=(10, 8))
disp_test.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for DL')
plt.savefig('Confusion_Matrix_Test.png')  # Save plot to file
plt.show()
