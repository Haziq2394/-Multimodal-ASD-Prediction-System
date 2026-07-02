import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

class DTModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, self.model.predict(X_train))
        print(f"✅ Decision Tree trained | Train Accuracy: {train_acc:.4f}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path="models/dt_model.pkl"):
        joblib.dump(self.model, path)
        print(f"✅ Decision Tree saved to {path}")

    def load(self, path="models/dt_model.pkl"):
        self.model = joblib.load(path)
        print(f"✅ Decision Tree loaded from {path}")


if __name__ == "__main__":
    X_dummy = np.random.randn(100, 26)
    y_dummy = np.random.randint(0, 2, 100)
    dt = DTModel()
    dt.train(X_dummy, y_dummy)
    proba = dt.predict_proba(X_dummy)
    print(f"✅ DT probability output shape: {proba.shape}")