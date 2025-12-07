# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from platform import python_version
import tensorflow as tf
from pyod.models.auto_encoder import AutoEncoder

# Load dataset
df = pd.read_csv('creditcard.csv')
print(df.head())

# Separate features and target
model_features = df.columns.drop('Class')
X = df[model_features]
y = df['Class']

print("Class distribution:\n", y.value_counts())
print("Feature matrix shape:", X.shape)

# AutoEncoder parameters
contamination = 0.5       # Expected proportion of outliers
epochs = 30               # Number of training epochs
hidden_neurons = [64, 30, 30, 64]

# Initialize and fit AutoEncoder
clf = AutoEncoder(
    contamination=0.5,
    hidden_neuron_list=[64, 30, 30, 64],     # your desired architecture
    epoch_num=30,                            # number of epochs
    batch_size=32,                          # optional: choose as you like
    hidden_activation_name='relu',
    dropout_rate=0.2,
    batch_norm=True
)

clf.fit(X)


# Predict outliers
outliers = clf.predict(X)
anomaly_indices = np.where(outliers == 1)[0]
print("Anomaly indices:", anomaly_indices)

# Example prediction on a sample
sample_idx = 4920
sample = X.iloc[[sample_idx]]
print("Sample features:\n", sample)
print("Sample actual class:", y.iloc[[sample_idx]])
print("Predicted class:", clf.predict(sample, return_confidence=False))
print("Prediction confidence:", clf.predict_confidence(sample))

# Get labels and decision scores
y_pred = clf.labels_  # Predicted labels for all data
y_scores = clf.decision_scores_  # Outlier scores (higher = more abnormal)

print("First 5 predicted labels:", y_pred[:5])
print("First 5 anomaly scores:", y_scores[:5])

# Plot anomaly scores with model threshold
plt.figure(figsize=(15, 8))
plt.plot(y_scores, label='Anomaly Scores')
plt.axhline(y=clf.threshold_, color='r', linestyle='dotted', label='Threshold')
plt.xlabel('Instances')
plt.ylabel('Anomaly Scores')
plt.title('Anomaly Scores with Auto-Calculated Threshold')
plt.legend()
plt.show()

# Plot anomaly scores with custom threshold
custom_threshold = 50
plt.figure(figsize=(15, 8))
plt.plot(y_scores, color='green', label='Anomaly Scores')
plt.axhline(y=custom_threshold, color='r', linestyle='dotted', label='Custom Threshold')
plt.xlabel('Instances')
plt.ylabel('Anomaly Scores')
plt.title('Anomaly Scores with Modified Threshold')
plt.legend()
plt.show()

# Plot training loss history
plt.figure(figsize=(15, 8))
pd.DataFrame(clf.history_).plot(title='AutoEncoder Training Loss')
plt.show()

# Scatter plot of transactions with anomaly scores
plt.figure(figsize=(15, 8))
sns.scatterplot(
    x='Time',
    y='Amount',
    hue=y_scores,
    size=y_scores,
    palette='RdBu_r',
    data=df,
    legend='full'
)
plt.xlabel('Time (seconds elapsed from first transaction)')
plt.ylabel('Transaction Amount')
plt.title('Transaction Scatter Plot Colored by Anomaly Scores')
plt.legend(title='Anomaly Scores')
plt.show()
