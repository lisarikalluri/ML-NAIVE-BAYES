# Import necessary libraries 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('EnjoySport.csv')
# Separate features and target variable
 
label_encoders = {}
for column in df.columns:
if df[column].dtype == 'object': 
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column]) 
    label_encoders[column] = le
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier 
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set 
y_pred = nb_classifier.predict(X_test)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
 
error_rate = 1 - accuracy
class_report = classification_report(y_test, y_pred)

# Display performance metrics print("Confusion Matrix:") print(conf_matrix) print("\nClassification Report:") print(class_report)
print(f"Accuracy: {accuracy * 100:.2f}%")
 print(f"Error Rate: {error_rate * 100:.2f}%")

# Plot the confusion matrix plt.figure(figsize=(4,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples", xticklabels=df['PlayTennis'].unique(), yticklabels=df['PlayTennis'].unique())
plt.xlabel("Predicted Labels") 
plt.ylabel("True Labels") 
plt.title("Confusion Matrix")
plt.show()
