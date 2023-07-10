import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mp

from sklearn import metrics
from sklearn.metrics import cohen_kappa_score as MC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Ridge
import itertools

# Data analysis
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#%matplotlib inline
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.dummy import DummyClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Ensemble Models
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Score functions
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Clustering and Principal Component Analysis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv(r'C:\Users\sophi\Desktop\Αριστοτέλειο\Μηχανική Μάθηση\Project 10\HTRU_2.csv')

data.head()

cols = [
    "Mean of the integrated profile",
    "Standard Deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard Deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve",
    "Class",
]

data.columns = cols

# Remove the target variable
y_data = data["Class"]
X_data = data.drop("Class", axis=1)

X_data.head()

print(f"The dataset has a shape (rows, cols) of {X_data.shape}!")

plt.figure(figsize=(13, 20))
for i, col in enumerate(X_data.columns):
    plt.subplot(4, 2, i + 1)
    sns.distplot(X_data[col], kde=False)
    plt.axvline(X_data[col].mean(), linestyle="dashed", label="Mean")
    plt.axvline(
        X_data[col].std(), color="b", linestyle="dotted", label="Standard Deviation"
    )
    plt.legend(loc="best")

# Visual representation of the statistical summary of data
# Boxenplot shows a large number of quantiles
plt.figure(figsize=(15, 25))
for i, col in enumerate(X_data.columns):
    plt.subplot(4, 2, i + 1)
    plt.savefig(col + ".png")
    sns.boxenplot(x=y_data, y=X_data[col])
    plt.title(col)

heat_map = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(heat_map, annot=True, linewidth=3)
plt.title("Correlation between Variables", fontsize=20)

# Split the dataset into a train test and a test set
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)

# Fit the StandardScaler to the train set
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_data.columns)

# Transform the test set
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_data.columns)

# Creating a dummy classifier
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_y_pred = dummy.predict(X_test)

# Dummy classifier performance
print('recall=',metrics.recall_score(y_test, dummy_y_pred, average='weighted',labels=np.unique(dummy_y_pred)))
print('precision=',metrics.precision_score(y_test, dummy_y_pred, average='weighted',labels=np.unique(dummy_y_pred)))
print('accuracy=',metrics.accuracy_score(dummy_y_pred,y_test))
print('f1=',metrics.f1_score(y_test, dummy_y_pred, average='weighted',labels=np.unique(dummy_y_pred)))

def print_metrics(y_test, y_pred):

    print("\033[1mMatthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))

    print("\nClassification report:\n", (classification_report(y_test, y_pred)))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        linecolor="k",
        linewidths=3,
    )
    plt.title("Confusion Matrix", fontsize=20)

performance = pd.DataFrame(index=["MCC", "F1 Score"])

# Fitting the model and predicting
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
# A glance at the performance (different metrics)
print_metrics(y_test, log_reg_pred)
# Keep model's performance for later comparison
performance["Logistic Regression"] = (matthews_corrcoef(y_test, log_reg_pred),f1_score(y_test, log_reg_pred),)

# Fitting the model and predicting
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
# A glance at the performance (different metrics)
print_metrics(y_test, knn_pred)
# Keep model's performance for later comparison
performance["K-Nearest Neighbors"] = (matthews_corrcoef(y_test, knn_pred),f1_score(y_test, knn_pred),)

# Fitting the model and predicting
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
# A glance at the performance (different metrics)
print_metrics(y_test, tree_pred)
# Keep model's performance for later comparison
performance["Decision Tree"] = (matthews_corrcoef(y_test, tree_pred),f1_score(y_test, tree_pred),)

# Fitting the model and predicting
bayes = GaussianNB()
bayes.fit(X_train, y_train)
bayes_pred = bayes.predict(X_test)
# A glance at the performance (different metrics)
print_metrics(y_test, bayes_pred)
# Keep model's performance for later comparison
performance["Naïve Bayes"] = (matthews_corrcoef(y_test, bayes_pred),f1_score(y_test, bayes_pred),)

# Fitting the model and predicting
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
# A glance at the performance (different metrics)
print_metrics(y_test, svc_pred)
# Keep model's performance for later comparison
performance["SVC"] = matthews_corrcoef(y_test, svc_pred), f1_score(y_test, svc_pred)


def performance_plot(performance):
    """Barplots of each model's performance on the Matthews Correlation Coefficient (MCC) and the F1 Score"""
    fig, ax = plt.subplots(figsize=(15, 7))
    w = 0.4
    i = np.arange(len(performance.columns))
    models = performance.columns

    bars_left = ax.bar(i - w / 2, performance.loc["MCC"] * 100, width=w, label="MCC")
    bars_right = ax.bar(
        i + w / 2, performance.loc["F1 Score"] * 100, width=w, label="F1 Score"
    )

    ax.grid(axis="y")
    ax.set_xticks(i)
    ax.set_xticklabels(models)
    ax.set_ylabel("Ratio (%)")
    ax.legend(loc="best")
    ax.set_ylim(0, 100)
    ax.set_title("Models performance")
    for bar in itertools.chain(bars_left, bars_right):
        height = bar.get_height()
        ax.annotate(
            "{:4.1f}".format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )

performance_plot(performance)

# Fitting the model and predicting
rand_forest = RandomForestClassifier(n_jobs=-1)
rand_forest.fit(X_train, y_train)
rand_forest_pred = rand_forest.predict(X_test)

# A glance at the performance (different metrics)
print_metrics(y_test, rand_forest_pred)

# Keep model's performance for later comparison
performance["Random Forest"] = (
    matthews_corrcoef(y_test, rand_forest_pred),
    f1_score(y_test, rand_forest_pred),
)

# Fitting the model and predicting
tree = DecisionTreeClassifier(criterion="entropy", max_depth=1)

adaboost = AdaBoostClassifier(
    base_estimator=tree, n_estimators=500, learning_rate=0.1, random_state=42
)
adaboost.fit(X_train, y_train)
adaboost_pred = adaboost.predict(X_test)

# A glance at the performance (different metrics)
print_metrics(y_test, adaboost_pred)

# Keep model's performance for later comparison
performance["AdaBoost"] = (
    matthews_corrcoef(y_test, adaboost_pred),
    f1_score(y_test, adaboost_pred),
)

# Fitting the model and predicting
estimators = [
    ("knn", knn),
    ("tree", tree),
    ("logreg", log_reg),
    ("svc", svc),
    ("nb", bayes),
]
stack = StackingClassifier(estimators=estimators)
stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)
# A glance at the performance (different metrics)
print_metrics(y_test, stack_pred)

# Keep model's performance for later comparison
performance["Stacking"] = (
    matthews_corrcoef(y_test, stack_pred),
    f1_score(y_test, stack_pred),
)

# Plotting each model's performances
performance_plot(performance)

X_for_pca = X_train.append(X_test, ignore_index=True)

# Principal Component Analysis with the 4 dimensions (features) of the dataset
pca = PCA(n_components=4)
pca.fit_transform(X_for_pca)

# We can now retrieve the important information
print(pca.explained_variance_ratio_)

print_metrics(y_test, stack_pred)



#Result: Precision metric is better using the most important features, as well as f1 when using the macro avg, while recall drops only a bit (0.02).