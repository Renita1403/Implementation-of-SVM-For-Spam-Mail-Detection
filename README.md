# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the spam.csv dataset using pandas
2. Select the message and label columns, then convert labels (ham = 0, spam = 1)
3. Convert the text messages into numerical features using TF-IDF vectorization
4. Split the dataset into training and testing data, then train the SVM classifier using the training set
5. Predict the test results, generate the confusion matrix, and display it using a heatmap graph

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Renita Ireen G
RegisterNumber:  212225220083

# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Create dataset
data = {
    'v1': ['ham','ham','spam','ham','ham'],
    'v2': [
        'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
        'Ok lar... Joking wif u oni...',
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
        'U dun say so early hor... U c already then say...',
        'Nah I don\'t think he goes to usf, he lives around here though'
    ]
}

df = pd.DataFrame(data)

# Encode labels (ham=0, spam=1)
df['label'] = df['v1'].map({'ham':0, 'spam':1})

# Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['v2'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict new message
new_message = ["Congratulations! You have won a free ticket to Bahamas. Call now!"]
new_message_vect = vectorizer.transform(new_message)
prediction = svm_model.predict(new_message_vect)
print(f"Prediction: {'Spam' if prediction[0]==1 else 'Ham'}")
```

## Output:

<img width="647" height="314" alt="Screenshot 2026-05-25 133619" src="https://github.com/user-attachments/assets/4bb6c8f3-0c91-4545-b715-2471e1a47ad2" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
