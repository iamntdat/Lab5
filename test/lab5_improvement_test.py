
"""
Model improvement experiment:
- Use TF-IDF + MultinomialNB as an alternative baseline
- Compare metrics with the LogisticRegression baseline
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from src.models.text_classifier import TextClassifier

def make_synthetic_data():
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

def run_improvement():
    X, y = make_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf_nb = TextClassifier(vectorizer='tfidf', random_state=42)
    # Use MultinomialNB as classifier inside our wrapper
    from sklearn.naive_bayes import MultinomialNB
    clf_nb.fit(X_train, y_train, clf=MultinomialNB())
    results = clf_nb.evaluate(X_test, y_test, average='macro')
    print("Improved model (TF-IDF + MultinomialNB) results:")
    print(results)
    return results

if __name__ == "__main__":
    run_improvement()
