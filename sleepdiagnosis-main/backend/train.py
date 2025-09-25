import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from model import extract_features

DATA_DIR = "data"
MODEL_OUT = "saved_model.joblib"

def gather_features(data_dir):
    X, y = [], []
    classes = {"normal": 0, "apnea": 1}
    for label in classes:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder): continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                continue
            path = os.path.join(folder, fname)
            try:
                feat = extract_features(path)
                X.append(feat["vector"])
                y.append(classes[label])
            except Exception as e:
                print(f"Skipping {path}: {e}")
    return np.vstack(X), np.array(y)

def main():
    X, y = gather_features(DATA_DIR)
    print("Shapes:", X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    print("Train acc:", clf.score(X_train, y_train))
    print("Test acc:", clf.score(X_test, y_test))
    joblib.dump(clf, MODEL_OUT)
    print("Saved model:", MODEL_OUT)

if __name__ == "__main__":
    main()

