# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:58:53 2024

@author: JSR
"""

import pandas as pd
import os
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

output_path = os.path.abspath(os.path.dirname(__file__))
data = pd.read_csv(os.path.join(output_path, 'data.csv')) # file should be saved in the same directory as the script

# Convert date columns to datetime
data['BKG_DT'] = pd.to_datetime(data['BKG_DT'])
data['CK_IN_DT'] = pd.to_datetime(data['CK_IN_DT'])
data['CK_OUT_DT'] = pd.to_datetime(data['CK_OUT_DT'])

df = data.copy()

#Check amount of null values and drop if insignificant amount
df = df.dropna(how='all')
null_percent  = df.isnull().mean() * 100
print(null_percent)

df = df.dropna(subset=['LOC_DESC', 'RATE_SEGMENT', 'BKG_CHANNEL'])

# Calculate durations
df['Length_of_Stay'] = (df['CK_OUT_DT'] - df['CK_IN_DT']).dt.days
df['Booking_Lead_Time'] = (df['CK_IN_DT'] - df['BKG_DT']).dt.days

# Extract day
df['BKG_Day'] = df['BKG_DT'].dt.dayofweek  # Monday=0, Sunday=6
df['CK_IN_Day'] = df['CK_IN_DT'].dt.dayofweek
df['CK_OUT_Day'] = df['CK_OUT_DT'].dt.dayofweek

# Extract Month
df['BKG_Month'] = df['BKG_DT'].dt.month
df['CK_IN_Month'] = df['CK_IN_DT'].dt.month
df['CK_OUT_Month'] = df['CK_OUT_DT'].dt.month

# Add seasons of booking
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'
df['BKG_Season'] = df['BKG_Month'].apply(get_season)

# Apply one-hot encoding for non-ordinal features LOC_DESC and BKG_CHANNEL
df = pd.get_dummies(df, columns=['LOC_DESC', 'BKG_CHANNEL', 'BKG_Season'])

# Apply label encoding for RATE_SEGMENT, assuming that the higher the number, the higher the value
label_encoder = LabelEncoder()
df['RATE_SEGMENT_encoded'] = label_encoder.fit_transform(df['RATE_SEGMENT'])

df = df.drop(['BKG_DT', 'CK_IN_DT', 'CK_OUT_DT', 'RATE_SEGMENT', 'RM_QTY', 'LOC_DESC_LOC_2', 'BKG_CHANNEL_CHANNEL_3'], axis=1)

# Convert STAY_PURPOSE (target variable) to binary (BUSINESS = 1, LEISURE = 0)
df['STAY_PURPOSE'] = df['STAY_PURPOSE'].apply(lambda x: 1 if x == 'BUSINESS' else 0)

# Check class distribution before splitting
print("Class distribution:\n", df['STAY_PURPOSE'].value_counts())

# Split the data into train and test sets with stratification
X = df.drop('STAY_PURPOSE', axis=1)
y = df['STAY_PURPOSE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Recheck class distribution in training and testing datasets
print("Train distribution:\n", y_train.value_counts())
print("Test distribution:\n", y_test.value_counts())

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Feature importance
feature_importances = clf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.ylabel('')
plt.xlabel('Feature Importance')
plt.savefig('feature.png', dpi=500, bbox_inches='tight')
plt.show()

# Predictions using the model
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Classification report
print("Classification report:", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6), dpi=500)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
             xticklabels=['Leisure', 'Business'], 
             yticklabels=['Leisure', 'Business'])
plt.text(0.5, 0.6, 'True Leisure', ha='center', va='center', color='white')
plt.text(1.5, 0.6, 'False Business', ha='center', va='center', color='black')
plt.text(0.5, 1.6, 'False Leisure', ha='center', va='center', color='black')
plt.text(1.5, 1.6, 'True Business', ha='center', va='center', color='black')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=500, bbox_inches='tight')
plt.show()

# ROC curve
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score: {roc_auc:.2f}")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6), dpi=500)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right", frameon=False)
plt.savefig('ROC.png', dpi=500, bbox_inches='tight')
plt.show()

# Analyze predictions to check distribution
predicted_counts = pd.Series(y_pred).value_counts(normalize=True) * 100
predicted_counts.index = ['Leisure', 'Business']
actual_distribution = {'Leisure': 68, 'Business': 32}
comparison_df = pd.DataFrame({
    'Predicted': predicted_counts,
    'Actual': actual_distribution.values()
}, index=actual_distribution.keys())
print("\nPredicted vs Actual Stay Purpose:", comparison_df)

comparison_df.plot(kind='bar', figsize=(8, 6), color=['#00008B', '#ADD8E6'])
plt.title('Predicted vs Actual Stay Purpose Distribution')
plt.ylabel('Percentage')
plt.xlabel('') 
plt.xticks(rotation=0)
plt.savefig('distribution.png', dpi=500, bbox_inches='tight')
plt.show()

# Saving model
model_filename = 'stay_purpose_randomforestmodel.joblib'
joblib.dump(clf, model_filename)