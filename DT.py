# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 21:10:55 2024

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#Loading data
encoded_data = pd.read_csv(r'~/Desktop/BRCA_project/results/encoded_data.csv', header=None)
encoded_data = encoded_data.astype(float)
multi = pd.read_csv(r"~/Desktop/BRCA_project/Processed/Multi_code.csv", index_col=0)
inter = pd.read_csv(r'~/Desktop/BRCA_project/results/intermediate.csv', header=None)
inter = inter.astype(float)
#Split the data into features and labels(change between intermediate and early data) 
X = encoded_data.values #inter.values 
y = multi['Multi_code'].values

#One-hot encode the labels
lb = LabelBinarizer()
y = lb.fit_transform(y)

#Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)

model = RandomForestClassifier(n_estimators=1000, random_state=42)

pipeline = Pipeline([
    ('smote', smote),
    ('classifier', model)
])

pipeline.fit(X_train, np.argmax(y_train, axis=1))

y_test_pred_proba = pipeline.predict_proba(X_test)
y_test_pred_classes = np.argmax(y_test_pred_proba, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

#Compute metrics on test data
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
plt.title('Confusion Matrix for DT')
plt.savefig('Confusion_Matrix_Test.png')  # Save plot to file
plt.show()
