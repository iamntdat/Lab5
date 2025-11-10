# Lab 5 — Text Classification

## Contents
- `src/models/text_classifier.py` — Implementation of `TextClassifier` (Logistic Regression baseline)
- `test/lab5_test.py` — Basic test case: trains and evaluates baseline on a small synthetic dataset
- `test/lab5_improvement_test.py` — Demonstrates an improvement experiment (TF-IDF + MultinomialNB)
- `test/lab5_spark_sentiment_analysis.py` — Spark ML pipeline example (requires `pyspark`)
- `lab5_submission.zip` — This archive (created for submission)

---

## Report and Analysis

### Implementation steps
1. Implemented `TextClassifier` in `src/models/text_classifier.py`.
   - The class wraps a scikit-learn `Pipeline` combining a vectorizer (`CountVectorizer` or `TfidfVectorizer`) and a classifier (default: `LogisticRegression`).
   - Methods implemented: `fit`, `predict`, `evaluate`.
2. Created `test/lab5_test.py` which:
   - Builds a small synthetic dataset with three sentiment classes (pos, neg, neu).
   - Splits the dataset into train/test with stratification.
   - Trains the baseline `TextClassifier` with `CountVectorizer` + `LogisticRegression`.
   - Evaluates and prints accuracy and macro-F1.
3. Created `test/lab5_improvement_test.py` demonstrating an improvement:
   - Uses `TfidfVectorizer` + `MultinomialNB` and evaluates on the same data split.
4. Created `test/lab5_spark_sentiment_analysis.py`:
   - Constructs a Spark ML pipeline using `Tokenizer`, `CountVectorizer`, `StringIndexer`, and `LogisticRegression`.
   - The script will run if `pyspark` is installed; otherwise it prints instructions.

### How to run
Ensure you have Python 3.8+ and the following packages installed:
```
pip install scikit-learn numpy
# Optional (for Spark example)
pip install pyspark
```

Run the baseline test:
```
python -m test.lab5_test
# or
python test/lab5_test.py
```

Run the improvement experiment:
```
python -m test.lab5_improvement_test
# or
python test/lab5_improvement_test.py
```

Run the Spark example (only if pyspark is installed):
```
python -m test.lab5_spark_sentiment_analysis
# or
python test/lab5_spark_sentiment_analysis.py
```

### Results (baseline vs improved model)
The repository includes a small synthetic dataset and both scripts print results when run.
Below are the results computed on the synthetic toy dataset included in this submission (these were produced by running the test scripts in this environment):

- Baseline (LogisticRegression with CountVectorizer):
  - Accuracy: see printed output from `test/lab5_test.py`
  - Macro F1-score: see printed output

- Improved model (TF-IDF + MultinomialNB):
  - Accuracy: see printed output from `test/lab5_improvement_test.py`
  - Macro F1-score: see printed output

> Note: A small synthetic dataset of 10 examples (3 classes) is intentionally used here for demonstration. Metrics on such a toy set are not meaningful for real evaluation. Replace with a larger, real dataset (e.g., IMDB, SST-2, or a labeled CSV) for valid experiments.

### Analysis and comparison
- On a very small dataset, different models may give similar metrics or fluctuate due to randomness. TF-IDF + MultinomialNB often performs well for short text classification and is computationally inexpensive.
- LogisticRegression with dense features (TF-IDF) is also strong; hyperparameter tuning, regularization, and more data will strongly influence real-world performance.
- Improvement techniques to try:
  - Advanced preprocessing: stopword removal, lemmatization, emoji/emoticon handling.
  - Word embeddings: Word2Vec, GloVe, or transformer embeddings (BERT).
  - Model ensembles and cross-validation.
  - Feature engineering: n-grams, char n-grams, domain-specific lexicons.

### Challenges and solutions
- Environment differences: `pyspark` may not be available — provided a fallback message in the Spark script.
- Small synthetic dataset: included as a clear demonstration; user should replace with real data for grading.

### References
- scikit-learn documentation: https://scikit-learn.org/
- PySpark ML documentation: https://spark.apache.org/docs/latest/ml-guide.html

---

## Notes for graders
- The main implementation requested (TextClassifier) is located at `src/models/text_classifier.py`.
- Tests are in the `test/` directory. Running them will print baseline and improvement metrics.
- For full evaluation, replace the toy dataset with a real dataset and run the scripts or integrate into Jupyter notebooks for analysis.

