# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:22:28 2024

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
#Loading early and intermediate bottleneck results
encoded_data = pd.read_csv(r'~/Desktop/BRCA_project/results/encoded_data.csv', header=None)
encoded_data = encoded_data.astype(float)
multi = pd.read_csv(r"~/Desktop/BRCA_project/Processed/Multi_code.csv", index_col=0)
inter = pd.read_csv(r'~/Desktop/BRCA_project/results/intermediate.csv', header=None)
inter = inter.astype(float)
#Split the data into features and labels (change between intermediate and early data)
X = inter.values #encoded_data.values 
y = multi['Multi_code'].values

#One-hot encode the labels 
lb = LabelBinarizer()
y = lb.fit_transform(y)

#Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

#Applying SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, np.argmax(y_train, axis=1))

svm_model = SVC(kernel='poly', probability=True, random_state=42)
svm_model.fit(X_train_smote, y_train_smote)

y_val_pred_proba = svm_model.predict_proba(X_val)
y_val_pred_classes = np.argmax(y_val_pred_proba, axis=1)
y_val_true_classes = np.argmax(y_val, axis=1)

#Compute metrics on validation data
val_auc = roc_auc_score(y_val, y_val_pred_proba, multi_class='ovr')
val_f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='weighted')
val_f1_macro = f1_score(y_val_true_classes, y_val_pred_classes, average='macro')
val_precision = precision_score(y_val_true_classes, y_val_pred_classes, average='weighted')
val_recall = recall_score(y_val_true_classes, y_val_pred_classes, average='weighted')

print(f'Validation AUC: {val_auc:.4f}')
print(f'Validation F1 Score (Weighted): {val_f1:.4f}')
print(f'Validation F1 Score (Macro): {val_f1_macro:.4f}')
print(f'Validation Precision: {val_precision:.4f}')
print(f'Validation Recall: {val_recall:.4f}')

#Evaluate on test data
y_test_pred_proba = svm_model.predict_proba(X_test)
y_test_pred_classes = np.argmax(y_test_pred_proba, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

#Compute metrics for test data
test_auc = roc_auc_score(y_test, y_test_pred_proba, multi_class='ovr')
test_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='weighted')
test_f1_macro = f1_score(y_test_true_classes, y_test_pred_classes, average='macro')
test_precision = precision_score(y_test_true_classes, y_test_pred_classes, average='weighted')
test_recall = recall_score(y_test_true_classes, y_test_pred_classes, average='weighted')

print(f'Test AUC: {test_auc:.4f}')
print(f'Test F1 Score (Weighted): {test_f1:.4f}')
print(f'Test F1 Score (Macro): {test_f1_macro:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')

#Confusion Matrix for Test Data
conf_matrix_test = confusion_matrix(y_test_true_classes, y_test_pred_classes)
disp_test = ConfusionMatrixDisplay(conf_matrix_test, display_labels=lb.classes_)

plt.figure(figsize=(10, 8))
disp_test.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for SVM')
plt.savefig('Confusion_Matrix_Test.png')  # Save plot to file
plt.show()
