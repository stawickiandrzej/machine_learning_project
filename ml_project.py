from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import time
import types
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

del sys.modules["os"]


dir_path = os.path.join("D:\\", "machine_learning_project")
train_data_file = "train_data.csv"
train_labels_file = "train_labels.csv"
test_data_file = "test_data.csv"
header_list = []


def loaddata(
    dir_path: str, file_name: str, header_list: list
) -> pd.core.frame.DataFrame:
    """
    The function loads the given file and returns it as DataFrame
    Args:
        dir_path:   application working directory
        file_name:  the name of file to be loaded
        header_list:list of column names

    Returns:

    """
    return pd.read_csv(os.path.join(dir_path, file_name), names=header_list)


def dump_file(_dir_path: str, _file_name: str, _buffer: pd.core.frame.DataFrame):
    """
    The function saves the given variable buffer into binary file
    Args:
        _dir_path:   application working directorty
        _file_name:  the name of file to be saved
        _buffer:     variable to write

    Returns:

    """
    with open(os.path.join(_dir_path, _file_name), "wb") as f:
        dump(_buffer, f)


def load_file(_dir_path: str, _file_name: str) -> pd.core.frame.DataFrame:
    """
    The function load dataset from binary file, and return pandas DataFrame
    Args:
        _dir_path:   application working directory
        _file_name:  binary file to load

    Returns: DataFrame

    """
    with open(os.path.join(_dir_path, _file_name), "rb") as f:
        return load(f)


def standard_scaler(_df: pd.core.frame.DataFrame) -> np.ndarray:
    """
    standardization of dataset data using StandardScaler
    Args:
        _df:    dataset to standarization

    Returns: Standardized features

    """
    scaler = StandardScaler().fit(_df)
    return scaler.transform(_df)


def split_dataset(
    _dff: pd.core.frame.DataFrame, _dft: pd.core.frame.DataFrame
) -> pd.core.frame.DataFrame:
    """
    splits the data into a test and training set
    Args:
        _dff:    DataFrame with features
        _dft:    DataFrame with targets

    Returns:    return train_features, train_targets, test_features, train_targets

    """
    return train_test_split(_dff, _dft, test_size=0.25, random_state=42, shuffle=True)


def min_max_scaler(_df: pd.core.frame.DataFrame) -> np.ndarray:
    """
    standardization of dataset data using MinMaxScaler
    Args:
        _df:    dataset to standarization

    Returns: Standardized features

    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(_df)


def log_transformation(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    # replace outliers with log_transformation
    for col in df.columns:
        df[str(col)].map(lambda i: np.log(i) if i > 0 else 0)
    return df


if __name__ == "__main__":
    start_time = time.time()
    # Load labels with list of headers into pandas datafame
    header_list = ["T0000"]
    df_targets = loaddata(dir_path, train_labels_file, header_list)

    # Make dataset header
    header_list = []
    for i in range(10000):
        header_list.append(f"F{i:04d}")

    # Load dataset with list of headers into pandas dataframe
    df_features = loaddata(dir_path, train_data_file, header_list)
    print(f"loading csv elapsed time: {time.time() - start_time}")

    start_time = time.time()
    np_features = min_max_scaler(df_features)
    print(f'oryginal shape {np_features.shape}')
    pca = PCA(n_components=0.99, svd_solver=auto)
    pca.fit(np_features)
    np_features = pca.transform(np_features)
    print(f"reducing dataset size elapsed time: {time.time() - start_time}")

    print(f'new shape {np_features.shape}')
    print(pca.explained_variance_ratio_.shape)
    (
        df_features_train,
        df_features_test,
        df_targets_train,
        df_targets_test,
    ) = split_dataset(df_features, df_targets)

    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, 3558, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0, 1.1)

    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
#    plt.xticks(np.arange(0, 11, step=350)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.text(0.5, 0.85, '99% cut-off threshold', color='red', fontsize=16)

    ax.grid(axis='x')
    plt.show()


log_reg_model = LogisticRegression()
svc_model = SVC()

pipe = Pipeline([("classifier", SVC())])

search_space = [
    {"classifier": [log_reg_model],
     "classifier__solver": ['saga'],
     "classifier__penalty": ['l2', 'elasticnet', "none"],
     "classifier__class_weight": ["balanced", None],
     "classifier__C": np.logspace(0, 4, 10),
     "classifier__multi_class": ['ovr']},

    {"classifier": [svc_model],
     "classifier__kernel":["linear", "rbf", "poly"],
     "classifier__class_weight": [None, "balanced"],
     "classifier__gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
     "classifier__C": np.logspace(0, 4, 10)},
]

gridsearch = GridSearchCV(pipe, search_space, cv=3, verbose=2, n_jobs=-1)

best_model = gridsearch.fit(df_features_train, df_targets_train)

print(best_model.best_estimator_.get_params()["classifier"])

print(gridsearch.best_params_)

model = LogisticRegression(C=1, class_weight=None,
                           multi_class='ovr', penalty='l2', solver='saga')

prediction = model.predict(df_features_test)

print(classification_report(df_targets_test, prediction))
