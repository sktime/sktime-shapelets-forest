import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from shapeletForest.ensemble import ShapeletForestClassifier
from sktime.datasets import load_gunpoint

if __name__ == "__main__":

    X_train, y_train = load_gunpoint(split='TRAIN', return_X_y=True)
    X_test, y_test = load_gunpoint(split='TEST', return_X_y=True)

    # In case the labels are not numbers
    if not isinstance(y_train[0], (int, float)):
        le = LabelEncoder()
        le.fit(np.unique(y_train))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
    else:
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

    f = ShapeletForestClassifier(n_shapelets=1, metric="scaled_dtw", metric_params={"r": 0.1})
    c = time.time()
    f.fit(X_train, y_train)
    print("Classes: ", f.classes_)
    print("Num dimensions: ", X_train.shape[1])
    print("Num patterns: ", X_train.shape[0])
    print("Num timepoints: ", len(X_train.iloc[0].iloc[0]))
    print("Accuracy: ", f.score(X_test, y_test))
    print("Time spent: ", round(time.time() - c))


