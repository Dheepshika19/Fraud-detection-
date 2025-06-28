import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------------------
# 📥 Load and Explore the Dataset
# ---------------------------------------

# Load dataset
data = pd.read_csv("creditcard.csv")

# Show first few rows
print(data.head())
print("\nShape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())

# Class distribution
print("\nFraudulent vs Non-Fraudulent Transactions:")
print(data['Class'].value_counts())

# ---------------------------------------
# 📊 Visualizations
# ---------------------------------------

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution (0 = Not Fraud, 1 = Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show(block=False)  # Keeps plot open and moves to next

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = data.corr()
sns.heatmap(corr, cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# ---------------------------------------
# 💡 Machine Learning Model: Logistic Regression
# ---------------------------------------

# Step 1: Split into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Step 2: Train-test split (with stratify for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🎯 Accuracy Score:", accuracy_score(y_test, y_pred))
