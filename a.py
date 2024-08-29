import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load the data
train_data = pd.read_csv('train_data.txt', delimiter=':::', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'], engine='python')

# Preprocess the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_data['DESCRIPTION'])

# Encode the target labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['GENRE'])

# Define models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate models
best_model_name = None
best_model = None
best_score = 0
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
    mean_score = scores.mean()
    print(f"{model_name} Cross-validation accuracy: {mean_score:.2f}")
    if mean_score > best_score:
        best_score = mean_score
        best_model_name = model_name
        best_model = model

print(f"\nBest Model: {best_model_name} with accuracy: {best_score:.2f}")

# Train the best model on the entire training data
best_model.fit(X_train_tfidf, y_train)

# Function to predict genre for a new description
def predict_genre(description):
    description_tfidf = vectorizer.transform([description])
    predicted_genre = best_model.predict(description_tfidf)
    return label_encoder.inverse_transform(predicted_genre)[0]

# Interactive input loop
while True:
    user_input = input("Enter a movie description (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    predicted_genre = predict_genre(user_input)
    print(f"Predicted Genre: {predicted_genre}")
