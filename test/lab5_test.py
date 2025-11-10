
"""
Basic test case:
- Build a small synthetic dataset
- Split into train/test
- Train TextClassifier (LogisticRegression with CountVectorizer)
- Evaluate and print metrics
"""
from sklearn.model_selection import train_test_split
from src.models.text_classifier import TextClassifier
from sklearn.datasets import fetch_20newsgroups
import random

def make_synthetic_data():
    # Small synthetic dataset with three classes: pos, neg, neutral
    X = [
        "I love this product, it works great and is amazing",
        "This is the worst thing I ever bought, awful experience",
        "It's okay, neither good nor bad",
        "Absolutely fantastic! Highly recommend it.",
        "Terrible, will never buy again.",
        "Mediocre performance but acceptable for the price.",
        "I am very happy with this purchase.",
        "I hate it. Broke after one day.",
        "Not bad, could be better.",
        "Exceeded my expectations, brilliant."
    ]
    y = ["pos","neg","neu","pos","neg","neu","pos","neg","neu","pos"]
    return X, y

def run_test():
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = TextClassifier(vectorizer='count', random_state=42)
    clf.fit(X_train, y_train)
    results = clf.evaluate(X_test, y_test, average='macro')
    print("Baseline (LogisticRegression with CountVectorizer) results:")
    print(results)
    return results

if __name__ == "__main__":
    run_test()
