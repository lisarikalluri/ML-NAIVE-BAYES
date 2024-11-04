# Import necessary libraries 

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
 
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the CSV file 
df = pd.read_csv('iris.csv')

print(df.columns)

X = df.drop('Species', axis=1) 
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train) 
y_pred = nb_classifier.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred) 
accuracy = accuracy_score(y_test, y_pred) 
error_rate = 1 - accuracy
class_report = classification_report(y_test, y_pred)

# Display performance metrics print("Confusion Matrix:") print(conf_matrix)
 
print("\nClassification Report:") 
print(class_report)
print(f"Accuracy: {accuracy * 100:.2f}%") 
print(f"Error Rate: {error_rate * 100:.2f}%") 
# Plot the confusion matrix plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=df['Species'].unique(), yticklabels=df['Species'].unique())
plt.xlabel("Predicted Labels") 
plt.ylabel("True Labels") 
plt.title("Confusion Matrix") 
plt.show()
