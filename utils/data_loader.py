from sklearn import datasets
import pandas as pd

def get_available_datasets():
    return [
        "Diabetes",
        "California Housing",
        "Wine Quality",
        "Breast Cancer",
        "Iris (Regression)"
    ]

def load_dataset(name):
    if name == "Diabetes":
        data = datasets.load_diabetes()
    elif name == "California Housing":
        data = datasets.fetch_california_housing()
    elif name == "Wine Quality":
        data = datasets.load_wine()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif name == "Iris (Regression)":
        data = datasets.load_iris()
    else:
        raise ValueError("Unsupported dataset")

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y
