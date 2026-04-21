import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def flatten_features(X: np.ndarray) -> np.ndarray:
    # X: (N, T, C) -> (N, T*C)
    return X.reshape(X.shape[0], -1)


def train_logreg(X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    """
    Strong baseline: Logistic Regression on flattened window signals.
    We still normalize with train stats already in dataset pipeline,
    but LR benefits from internal scaling too.
    """
    Xtr = flatten_features(X_train)

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(
            max_iter=3000,
            n_jobs=-1,
            random_state=seed
        ))
    ])
    clf.fit(Xtr, y_train)
    return clf


def predict_logreg(model, X: np.ndarray) -> np.ndarray:
    Xf = flatten_features(X)
    return model.predict(Xf)