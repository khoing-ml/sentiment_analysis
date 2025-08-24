# scripts/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature
from pathlib import Path

# Start an MLflow run
with mlflow.start_run():
    # 1. Load Data
    df = pd.read_csv('data/IMDB Dataset.csv')
    # Quick cleanup and label encoding
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    # 3. Create a model pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('model', LogisticRegression(C=1.0)) # We can tune this C parameter
    ])

    # 4. Train the model
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    preds = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy}")

    # 6. Log with MLflow (include params, metrics, signature & example)
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", accuracy)

    # Prepare input example (single-row DataFrame) & infer signature
    input_example = pd.DataFrame({"review": [X_test.iloc[0]]})
    signature = infer_signature(input_example, pipeline.predict([X_test.iloc[0]]))

    # Prefer new 'name' argument (Model Registry). Fallback to deprecated artifact_path if unsupported.
    try:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="sentiment-model",
            input_example=input_example,
            signature=signature,
        )
        print("Model logged with name=sentiment-model (registry).")
    except Exception as e:
        print(f"Registry log failed ({e}); falling back to artifact_path.")
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sentiment-model",
            input_example=input_example,
            signature=signature,
        )
        print("Model logged with artifact_path=sentiment-model.")

    # Persist run id for API to dynamically load latest model
    run_id = mlflow.active_run().info.run_id
    Path("latest_run_id.txt").write_text(run_id)
    print(f"Saved latest run id to latest_run_id.txt: {run_id}")

    print("Run complete. Check the MLflow UI.")