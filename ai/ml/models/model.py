import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False   # time-series safe
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n================ MODEL PERFORMANCE ================")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    return model
