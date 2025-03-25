from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize the Spark session
spark = SparkSession.builder.appName("AdvancedSocialMediaSentimentML").getOrCreate()

# Load the dataset with sentiment labels
data = spark.read.csv("social_media_large_dataset.csv", header=True, inferSchema=True)
print("Loaded Data:")
data.show(truncate=False)

# Stage 1: Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")

# Stage 2: Remove stop words
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Stage 3: Generate bi-grams (n-grams with n=2)
ngram = NGram(n=2, inputCol="filtered_words", outputCol="ngrams")

# Stage 4: Convert n-grams to numerical features with HashingTF
hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=2000)

# Stage 5: Rescale features using IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Stage 6: Convert sentiment strings into numeric labels with handleInvalid set to "keep"
label_indexer = StringIndexer(inputCol="sentiment", outputCol="label", handleInvalid="keep")

# Stage 7: Define the Logistic Regression classifier (the ML model used)
lr = LogisticRegression(maxIter=20)

# Build the ML pipeline
pipeline = Pipeline(stages=[tokenizer, stopwords_remover, ngram, hashingTF, idf, label_indexer, lr])

# Set up hyperparameter tuning with a parameter grid for Logistic Regression
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Define the evaluator using accuracy as the metric
evaluator = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")

# Set up CrossValidator to perform 3-fold cross-validation
crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

# Split the data into training and test sets
train, test = data.randomSplit([0.7, 0.3], seed=42)

# Train the model using cross-validation
cvModel = crossval.fit(train)

# Make predictions on the test set
predictions = cvModel.transform(test)

# Evaluate the model's accuracy
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.2f}")

# Display sample predictions
predictions.select("id", "text", "sentiment", "prediction").show(truncate=False)

# Stop the Spark session
spark.stop()
