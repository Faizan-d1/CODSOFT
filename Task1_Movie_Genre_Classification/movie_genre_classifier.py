import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD AND PREPARE DATA
import pandas as pd

genres = []
descriptions = []

with open("train_data.txt", "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(" ::: ")
        
        # We expect 4 parts: ID, Title, Genre, Description
        if len(parts) == 4:
            _, title, genre, description = parts
            genres.append(genre.strip())
            descriptions.append(description.strip())

data = pd.DataFrame({
    "genre": genres,
    "description": descriptions
})

print("📊 Dataset Loaded")
print("Total samples:", len(data))
print("Unique genres:", data["genre"].nunique())
print(data.head())
print()

# 2. SPLIT DATA
X = data["description"]
y = data["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print()

# 3. TEXT VECTORIZATION (TF-IDF)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)  

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. TRAIN MODEL
model = LinearSVC()
model.fit(X_train_tfidf, y_train)


# 5. EVALUATE MODEL
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", round(accuracy, 4))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))
