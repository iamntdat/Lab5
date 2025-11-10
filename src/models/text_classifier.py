
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class TextClassifier:
    """
    Simple text classifier wrapper following the lab's requirements.

    Methods
    -------
    fit(X, y, vectorizer='count', **clf_kwargs)
        Train a logistic regression model using a vectorizer.
    predict(X)
        Predict labels for X.
    evaluate(X, y, average='macro')
        Return dict with accuracy and f1-score.
    """
    def __init__(self, vectorizer='count', random_state=42):
        if vectorizer not in ('count', 'tfidf'):
            raise ValueError("vectorizer must be 'count' or 'tfidf'")
        self.vectorizer = vectorizer
        self.random_state = random_state
        self.pipeline = None

    def _make_pipeline(self, clf=None):
        if self.vectorizer == 'count':
            vec = CountVectorizer()
        else:
            vec = TfidfVectorizer()
        if clf is None:
            clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        return Pipeline([('vect', vec), ('clf', clf)])

    def fit(self, X, y, clf=None):
        """Train the classifier pipeline on X, y."""
        self.pipeline = self._make_pipeline(clf)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """Return predicted labels for X."""
        if self.pipeline is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        return self.pipeline.predict(X)

    def evaluate(self, X, y, average='macro'):
        """Compute accuracy and F1-score."""
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average=average)
        return {'accuracy': float(acc), 'f1_'+average: float(f1)}
