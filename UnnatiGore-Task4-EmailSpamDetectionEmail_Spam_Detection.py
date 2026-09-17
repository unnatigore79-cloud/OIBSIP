import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample Dataset
data = {
    'text': [
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
        "Nah I don't think he goes to usf, he lives around here though",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!",
        "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera Free!",
        "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?",
        "SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575."
    ],
    'label': ['spam', 'ham', 'spam', 'spam', 'ham', 'spam']
}

df = pd.DataFrame(data)

# 2. Vectorization & Split
cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
