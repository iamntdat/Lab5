
"""
Spark example script.
This script constructs a Spark ML pipeline for text classification using
HashingTF or CountVectorizer, StringIndexer, and LogisticRegression.
It will attempt to import pyspark and build/run a small pipeline if available.
If pyspark is not installed, the script prints instructions.
"""
try:
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF, StringIndexer
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import pyspark.sql.functions as F

    def run_spark_example():
        spark = SparkSession.builder.master("local[2]").appName("lab5_spark_example").getOrCreate()
        data = [
            (0, "I love this product, it works great and is amazing", "pos"),
            (1, "This is the worst thing I ever bought, awful experience", "neg"),
            (2, "It's okay, neither good nor bad", "neu"),
            (3, "Absolutely fantastic! Highly recommend it.", "pos"),
            (4, "Terrible, will never buy again.", "neg")
        ]
        df = spark.createDataFrame(data, ["id", "text", "label"])
        # StringIndexer for label
        indexer = StringIndexer(inputCol='label', outputCol='labelIndex')
        tokenizer = Tokenizer(inputCol='text', outputCol='words')
        vectorizer = CountVectorizer(inputCol='words', outputCol='features')
        lr = LogisticRegression(featuresCol='features', labelCol='labelIndex', maxIter=20)
        pipeline = Pipeline(stages=[indexer, tokenizer, vectorizer, lr])
        model = pipeline.fit(df)
        preds = model.transform(df)
        evaluator = MulticlassClassificationEvaluator(labelCol='labelIndex', predictionCol='prediction', metricName='accuracy')
        acc = evaluator.evaluate(preds)
        print("Spark pipeline example accuracy on toy data:", acc)
        spark.stop()

    if __name__ == "__main__":
        run_spark_example()

except Exception as e:
    print("pyspark is not available in this environment or an error occurred.")
    print("To run the Spark example locally, install pyspark and run this script.")
    print("Detailed error:", e)
