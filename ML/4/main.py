import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score,  roc_auc_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def replace_value(name):
    unique_genders = df[name].unique()
    gender_dict = {name: i for i, name in enumerate(unique_genders)}
    df[name] = df[name].replace(gender_dict)


def metric_calculation(y_pred_train, y_pred):
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    train_precision = precision_score(y_train, y_pred_train, average='micro')
    test_precision = precision_score(y_test, y_pred, average='micro')

    print(f"Train precision: {train_precision:.4f}")
    print(f"Test precision: {test_precision:.4f}")

    train_recall = recall_score(y_train, y_pred_train, average='micro')
    test_recall = recall_score(y_test, y_pred, average='micro')

    print(f"Train recall: {train_recall:.4f}")
    print(f"Test recall: {test_recall:.4f}")

    train_f1 = f1_score(y_train, y_pred_train, average='micro')
    test_f1 = f1_score(y_test, y_pred, average='micro')

    print(f"Train F1-score: {train_f1:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")

    mcc_train = matthews_corrcoef(y_train, y_pred_train)
    mcc_test = matthews_corrcoef(y_test, y_pred)

    print(f"Train Matthews Correlation Coefficient (MCC): {mcc_train:.4f}")
    print(f"Test Matthews Correlation Coefficient (MCC): {mcc_test:.4f}")

    ba_train = balanced_accuracy_score(y_train, y_pred_train)
    ba_test = balanced_accuracy_score(y_test, y_pred)

    print(f"Balanced Accuracy (BA) : {ba_train:.4f}")
    print(f"Balanced Accuracy (BA) : {ba_test:.4f}")

    auc_train = roc_auc_score(y_train, knn.predict_proba(X_train), multi_class='ovr')
    auc_test = roc_auc_score(y_test, knn.predict_proba(X_test), multi_class='ovr')

    print(f"Balanced Area Under Curve for Receiver Operation Curve : {auc_train:.4f}")
    print(f"Balanced Area Under Curve for Receiver Operation Curve : {auc_test:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('matrix configuration для тестової вибірки')
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("dataset3_l4.csv")

    object_columns = list(df.select_dtypes(include='object').columns)
    for i in object_columns:
        replace_value(i)

    # 1 task
    print(df.head())

    ## 2 task
    print('\nРозмір data frame:', df.shape[1])

    ## 3 task
    print('\nАтрибути набору даних:\n', df.columns)

    ## 4 task
    while True:
        n_splits = int(input("Введіть кількість перемішувань (не менше 3): "))
        if n_splits < 3:
            print("Помилка! введіть число не менше 3.")
        else:
            break

    ## відсоток тестової вибірки 20%
    ss = ShuffleSplit(n_splits=n_splits, test_size=0.2)
    print(ss)
    a = 0
    x_train, x_test = pd.DataFrame(), pd.DataFrame()
    for train_index, test_index in ss.split(df):
        if a == 1:
            x_train, x_test = df.iloc[train_index], df.iloc[test_index]
        a += 1

    bCount = df["NObeyesdad"].value_counts()

    list_balance = [i/sum(bCount) for i in bCount]
    plt.plot(list_balance)
    plt.ylim(0, 1)
    plt.xlabel('Номер класу')
    plt.ylabel('Кількість спостережень')
    plt.show()
    print("Так, як значення співідношень кількості кожного класу до загальної кількості спостережнь майже схожі між собою(різниця не велика),то можемо сказати, що дані збалансовані.\n" )

    #5 task
    X_train = x_train.drop(columns=["NObeyesdad"])
    y_train = x_train["NObeyesdad"]
    X_test = x_test.drop(columns=["NObeyesdad"])
    y_test = x_test["NObeyesdad"]
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)

    #6 task
    metric_calculation(y_pred_train, y_pred_test)

    #7 task
    X = df.drop(columns=["NObeyesdad"])
    y = df["NObeyesdad"]
    p_values = list(range(1, 21))
    accuracy_scores = []
    for p in p_values:
        knn = KNeighborsClassifier(n_neighbors=5, p=p)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        accuracy_scores.append(scores.mean())
    plt.plot(p_values, accuracy_scores)
    plt.xlabel('Степінь Мінковського')
    plt.ylabel('Точність класифікації')
    plt.title('Залежність точності від степеня Мінковського')
    plt.show()
    print('Чим більший степінь Мінковського, тим нижча точність класифікації.')

