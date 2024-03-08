import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Using the 'Agg' backend for non-interactive use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Load dataset
# Assuming 'data.csv' contains your dataset
dataset = pd.read_csv('heart.csv')

# Split data into features and target variable
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target
label_encoder = LabelEncoder()

# Encode target variable
y = label_encoder.fit_transform(y)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# List of classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "K Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "NuSVC": NuSVC(),
    "LinearSVC": LinearSVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "LightGBM": LGBMClassifier(),
    "MLP": MLPClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Bernoulli Naive Bayes": BernoulliNB(),
    "Ridge Classifier": RidgeClassifier(),
    "Passive Aggressive Classifier": PassiveAggressiveClassifier(),
    "Stochastic Gradient Descent": SGDClassifier(),
    "Nearest Centroid": NearestCentroid(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "Extra Tree Classifier": ExtraTreeClassifier()
    # Add more classifiers here
}

# Dictionary to store evaluation metrics
evaluation_metrics = {
    "Accuracy": accuracy_score,
    "F1 Score": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
    "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
}


# Training and evaluating classifiers
results = {}
for name, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    results[name] = {}
    for metric_name, metric_function in evaluation_metrics.items():
        results[name][metric_name] = metric_function(y_test, y_pred)

# Plotting evaluation metrics
plt.figure(figsize=(14, 8))
for metric_name in evaluation_metrics.keys():
    values = [result[metric_name] for result in results.values()]
    sns.barplot(x=list(results.keys()), y=values, palette='viridis')
    plt.xlabel('Classifier')
    plt.ylabel(metric_name)
    plt.title(f'Classifier Comparison based on {metric_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{metric_name}_comparison.png')
    plt.show()

# Plotting confusion matrices
plt.figure(figsize=(14, 8))
for name, classifier in classifiers.items():
    plt.subplot(3, 10, list(classifiers.keys()).index(name) + 1)
    sns.heatmap(confusion_matrix(y_test, classifier.predict(X_test)), annot=True, fmt='d', cmap='viridis')
    plt.title(name)
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
