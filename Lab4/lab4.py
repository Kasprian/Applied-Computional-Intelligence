import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def visualize_data(data: pd.DataFrame, x_series: pd.Series, y_series: pd.Series, hue_series: pd.Series = None, 
                size_series: pd.Series = None, col_series: pd.Series = None, palette:bool = False):
    
    # Define the column wrapper
    num_rows = None
    if col_series is not None:
        if col_series.value_counts().shape[0] > 2:
            num_rows = math.ceil(col_series.value_counts().shape[0]/2)

    # Generate palette
    if palette:
        col_pal=sns.color_palette("pastel", col_series.value_counts().shape[0])
        sns.set_palette(col_pal)

    # Generate Seaborn-plot
    sns.relplot(data = data, x = x_series, y = y_series, hue = hue_series,
                size = size_series, col = col_series, col_wrap = num_rows)
    plt.show()

def split_dataset(data: pd.DataFrame, train_percentage: int, random_state_seed: int = 10):
    # Assert that train_percentage is between 0 and 100
    assert train_percentage >= 0 and train_percentage <= 100, "Oopsie train-percentage must be between 0 and 100!"

    # Split data into datasets
    train_split, test_split = train_test_split(data, train_size = 0.01*train_percentage, random_state=random_state_seed)
    print("original data-shape:", data.shape, "\ntrain data-shape:", train_split.shape, "\ntest data-shape:", test_split.shape)

    return train_split, test_split

# FUNCTIONS TO DEFINE GAUSSIAN CLASSIFIER
# ======================================= 
def generate_gaussian_classifier(train_data: pd.DataFrame, target_label: str):
    # Separate data into targets and training-data
    target_series = train_data.loc[:, target_label]
    train_df = train_data.loc[:, train_data.columns != target_label]
    # Classify the gaussian classifier
    clf = GaussianNB()
    clf.fit(train_df, target_series)

    return clf

def test_gaussian_classifier(test_data: pd.DataFrame, classifier: GaussianNB, target_label: str):
    # Separate data into targets and test-data
    true_labels = test_data.loc[:, target_label]
    test_df = test_data.loc[:, test_data.columns != target_label]
    # Calculate the score on the test-set and true values
    score = classifier.score(test_df, true_labels)
    print("The Gaussian-classifier managed a mean-score of:", score, "on", target_label)

# FUNCTIONS TO DEFINE THE LINEAR-VECTOR CLASSIFIER
# ================================================ 
def generate_LSVC_classifier(train_data: pd.DataFrame, target_label: str):
    # Separate data into targets and training-data
    target_series = train_data.loc[:, target_label]
    train_df = train_data.loc[:, train_data.columns != target_label]
    # Classify the gaussian classifier
    clf = LinearSVC(dual = False)
    clf.fit(train_df, target_series)

    return clf

def test_LSVC_classifier(test_data: pd.DataFrame, classifier: LinearSVC, target_label: str):
    # Separate data into targets and test-data
    true_labels = test_data.loc[:, target_label]
    test_df = test_data.loc[:, test_data.columns != target_label]
    # Calculate the score on the test-set and true values
    score = classifier.score(test_df, true_labels)
    print("The LSVC-classifier managed a mean-score of:", score, "on", target_label)

# FUNCTIONS TO DEFINE THE SVM CLASSIFIER
# ================================================ 
def generate_SVM_classifier(train_data: pd.DataFrame, target_label: str, desired_kernel = 'rbf'):
    # Separate data into targets and training-data
    target_series = train_data.loc[:, target_label]
    train_df = train_data.loc[:, train_data.columns != target_label]
    # Classify the vector-machine classifier
    clf = svm.SVC(kernel = desired_kernel)
    clf.fit(train_df, target_series)

    return clf

def test_SVM_classifier(test_data: pd.DataFrame, classifier: svm.SVC, target_label: str):
    # Separate data into targets and test-data
    true_labels = test_data.loc[:, target_label]
    test_df = test_data.loc[:, test_data.columns != target_label]
    # Calculate the score on the test-set and true values
    score = classifier.score(test_df, true_labels)
    print("The SVM-classifier managed a mean-score of:", score, "on", target_label)

# FUNCTIONS TO DEFINE THE KNN CLASSIFIER
# ================================================ 
def generate_KNN_classifier(train_data: pd.DataFrame, target_label: str, num_neighbors = 3):
    # Separate data into targets and training-data
    target_series = train_data.loc[:, target_label]
    train_df = train_data.loc[:, train_data.columns != target_label]
    # Classify the KNN classifier
    clf = KNeighborsClassifier(n_neighbors = num_neighbors)
    clf.fit(train_df, target_series)

    return clf

def test_KNN_classifier(test_data: pd.DataFrame, classifier: KNeighborsClassifier, target_label: str):
    # Separate data into targets and test-data
    true_labels = test_data.loc[:, target_label]
    test_df = test_data.loc[:, test_data.columns != target_label]
    # Calculate the score on the test-set and true values
    score = classifier.score(test_df, true_labels)
    print("The KNN-classifier managed a mean-score of:", score, "on", target_label)



def main():
    # Define column_names
    iris_headers = ['sepal_l_cm', 'sepal_w_cm', 'petal_l_cm', 'petal_w_cm', 'label']
    haberman_headers = ['age', 'operation_year', 'pos. axillary nodes', 'label (survival)']

    # Load in the Iris and Haberman datasets
    iris_data = pd.read_csv("iris\iris\iris.data", names=iris_headers)
    haberman_data = pd.read_csv("haberman\haberman\haberman.data", names=haberman_headers)
    print("Iris data:\n", iris_data.head())
    print("Haberman data:\n", haberman_data.head())

    # Vizualize data
    visualize_data(iris_data, x_series=iris_data['sepal_l_cm'], y_series=iris_data['sepal_w_cm'],
                        hue_series=iris_data['label'], size_series=iris_data['petal_w_cm'])
    visualize_data(iris_data, x_series=iris_data['petal_l_cm'], y_series=iris_data['petal_w_cm'],
                        hue_series= iris_data['label'], col_series=iris_data['label'], palette=True)
    visualize_data(haberman_data, x_series = haberman_data['operation_year'], y_series = haberman_data['pos. axillary nodes'], 
                    hue_series = haberman_data['label (survival)'], size_series = haberman_data['age'])

    # Split dataset into two sets
    train_set_iris, test_set_iris = split_dataset(iris_data, train_percentage = 75, random_state_seed = 15)
    train_set_haberman, test_set_haberman = split_dataset(haberman_data, train_percentage=70, random_state_seed = 10)

    # Train gaussian classifiers
    print("\nGaussian classifiers:\n===========================")
    gaussian_clf_iris = generate_gaussian_classifier(train_set_iris, 'label')
    test_gaussian_classifier(test_set_iris, gaussian_clf_iris, 'label')

    gaussian_clf_haberman = generate_gaussian_classifier(train_set_haberman, 'label (survival)')
    test_gaussian_classifier(test_set_haberman, gaussian_clf_haberman, 'label (survival)')

    # Train LSVC classifiers
    print("\nLSVC classifiers:\n===========================")
    LSVC_clf_iris = generate_LSVC_classifier(train_set_iris, 'label')
    test_LSVC_classifier(test_set_iris, LSVC_clf_iris, 'label')

    LSVC_clf_Haberman = generate_LSVC_classifier(train_set_haberman, 'label (survival)')
    test_LSVC_classifier(test_set_haberman, LSVC_clf_Haberman, 'label (survival)')

    # Train SVM classifiers
    print("\nSVM classifiers:\n===========================")
    SVM_clf_iris = generate_SVM_classifier(train_set_iris, 'label', 'linear')
    test_SVM_classifier(test_set_iris, SVM_clf_iris, 'label')

    SVM_clf_Haberman = generate_SVM_classifier(train_set_haberman, 'label (survival)')
    test_SVM_classifier(test_set_haberman, SVM_clf_Haberman, 'label (survival)')

    # Train KNN classifiers
    print("\nKNN classifiers:\n===========================")

    KNN_clf_iris = generate_KNN_classifier(train_set_iris, 'label', num_neighbors = 7)
    test_KNN_classifier(test_set_iris, KNN_clf_iris, 'label')

    KNN_clf_Haberman = generate_KNN_classifier(train_set_haberman, 'label (survival)', num_neighbors=7)
    test_KNN_classifier(test_set_haberman, KNN_clf_Haberman, 'label (survival)')

    return

main()
# %%
