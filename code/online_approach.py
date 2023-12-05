import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def features(df):
    X = df['text']
    y = df['category']
    return X, y


def etc_model(clf, X_train_tfidf, X_test_tfidf, y_train, y_test, X_test):
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    y_preddf = pd.DataFrame(y_pred)
    y_preddf.rename(columns={0: 'category predicted'}, inplace=True)
    x_testdf = pd.DataFrame(X_test)
    y_testdf = pd.DataFrame(y_test)
    x_testdf['id'] = range(1, len(x_testdf) + 1)
    y_testdf['id'] = range(1, len(y_testdf) + 1)
    y_preddf['id'] = range(1, len(y_preddf) + 1)
    join1 = y_testdf.merge(x_testdf, how='inner', indicator=False)
    join_df = join1.merge(y_preddf, how='inner', indicator=False)
    print(join_df[20:30])
    accuracy_score = metrics.accuracy_score(y_pred, y_test) * 100
    return cm, accuracy_score


def plot(cm, figure_path):
    confusion_matrix = pd.DataFrame(cm, index=["ChatGPT", "Human"], columns=["ChatGPT", "Human"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt='g')
    plt.savefig(figure_path)


def model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    lg = LogisticRegression(penalty='l1', solver='liblinear')
    sv = SVC(kernel='sigmoid', gamma=1.0)
    mnb = MultinomialNB()
    dtc = DecisionTreeClassifier(max_depth=5)
    knn = KNeighborsClassifier()
    rfc = RandomForestClassifier(n_estimators=50, random_state=2)
    etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
    abc = AdaBoostClassifier(n_estimators=50, random_state=2)
    bg = BaggingClassifier(n_estimators=50, random_state=2)
    gbc = GradientBoostingClassifier(n_estimators=50, random_state=2)

    def prediction(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        pr = model.predict(X_test)
        acc_score = metrics.accuracy_score(y_test, pr)
        f1 = metrics.f1_score(y_test, pr, average="binary", pos_label="chatgpt")
        return acc_score, f1

    acc_score = {}
    f1_score = {}
    clfs = {
        'LR': lg,
        'SVM': sv,
        'MNB': mnb,
        'DTC': dtc,
        'KNN': knn,
        'RFC': rfc,
        'ETC': etc,
        'ABC': abc,
        'BG': bg,
        'GBC': gbc,
    }
    for name, clf in clfs.items():
        acc_score[name], f1_score[name] = prediction(clf, X_train_tfidf, X_test_tfidf, y_train, y_test)
    return acc_score, f1_score, X_train_tfidf, X_test_tfidf, y_train, y_test, X_test


def online(df, figure_path):
    print("In existing method")
    X, y = features(df)
    print("Existing method Feature Done")
    accuracy, f1_score, X_train_tfidf, X_test_tfidf, y_train, y_test, X_test = model(X, y)
    print("Existing method accuracy Done")
    etc_classifier = ExtraTreesClassifier(n_estimators=50, random_state=2)
    cm, accuracy_score = etc_model(etc_classifier, X_train_tfidf, X_test_tfidf, y_train, y_test, X_test)
    plot(cm, figure_path)
    print("Existing method plot done")
    return accuracy_score
