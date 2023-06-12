import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import *
import matplotlib.pyplot as plt
import random
import sklearn.metrics as skm

NAMEMETRICS = ['Accuracy', 'Precision', 'Recall', 'F-scores', 'MCC', 'BA', 'YJS', 'AUC for PRC', 'AUC for ROC']

print('------TASK 1-------')
df = pd.read_csv("D:\Downloads\KM-03-3.csv")
print(df)

print('------TASK 2-------')
bCount = df['GT'].value_counts()
#відносна похибка з рівнем значущості a=0.05
bRat = abs(bCount[0] - bCount[1])/max(bCount[0], bCount[1])
print(" К-сть об'єктів кожного класу:\n", bCount)
if bRat <= 0.05: print('Набір даних збалансований\n')
else: print('Набір даних не збалансований\n')

print('------TASK 3a-------')
thresholds = [round(x, 2) for x in np.arange(0.05, 1, 0.05)]


def calcMetrics(nummodel, a):
    #Accuracy metric
    aMetric = [accuracy_score(df["GT"], df[nummodel + str(x)]) for x in thresholds]

    #Precision metric
    pMetric = [precision_score(df["GT"], df[nummodel + str(x)]) for x in thresholds]

    #Recall metric
    rMetric = [recall_score(df["GT"], df[nummodel + str(x)]) for x in thresholds]

    #F-Scores metric
    fMetric = [f1_score(df["GT"], df[nummodel + str(x)]) for x in thresholds]

    #Matthews Correlation Coefficient metric
    mccMetric = [matthews_corrcoef(df["GT"], df[nummodel + str(x)]) for x in thresholds]

    #Balanced Accuracy metric
    baMetric = [balanced_accuracy_score(df["GT"], df[nummodel + str(x)]) for x in thresholds]
    print(len(baMetric))
    #Youden's J statistics metric
    yjMetric = []
    for i in range(0, int(len(thresholds))):
        yjMetric.append(rMetric[i]+baMetric[i]-1)

    #Area Under Curve for Precision-Recall Curve metric
    aucprMetric = []
    for x in thresholds:
        pres1, rec1, thres1 = precision_recall_curve(df["GT"], df[nummodel + str(x)], pos_label=1)
        aucprMetric.append(auc(rec1, pres1))

    #Area Under Curve for Receiver Operation Curve metric
    aucroMetric = [roc_auc_score(df["GT"], df[nummodel + str(x)]) for x in thresholds]

    print("  Metrics Model", a)
    print("Accuracy ", aMetric)
    print("Precision ", pMetric)
    print("Recall ", rMetric)
    print("F-scores ", fMetric)
    print("Matthews Correlation Coefficient ", mccMetric)
    print("Balanced accuracy ", baMetric)
    print("Youden's J statistics ", yjMetric)
    print("Area Under Curve for Precision-Recall Curve ", aucprMetric)
    print("Area Under Curve for Receiver Operation Curve ", aucroMetric)
    print("\n")

    return aMetric, pMetric, rMetric, fMetric, mccMetric, baMetric, yjMetric,  aucprMetric, aucroMetric


def plot3B(s):
    q = 0
    m = ' 1'
    u=0
    for i in s:
        if q == 8:
            m = ' 2'
            q = 0
            u = 1
        if u:
            plt.plot(thresholds, i, label=NAMEMETRICS[q]+m)
        else:
            plt.plot(thresholds, i, label=NAMEMETRICS[q]+m, linestyle="--")
        plt.scatter(thresholds[i.index(max(i))], max(i), color='red')
        q += 1
    plt.xlabel("Thresholds")
    plt.ylabel("Metrics")
    plt.title("Metrics of 2 models")
    plt.legend(loc='lower right', ncol=9, frameon=True)
    plt.show()


def c3(numModel, numClass, s):
    numModel2 = 'Model_' + str(numModel) + '_'
    nClass = [df[numModel2 + str(x)].value_counts()[numClass] for x in thresholds]

    q = 0
    if numModel == 2:s = s[9::]

    for i in s:
        if q == 9:
            break
        a = 1
        if q == 0: a = 5
        if q == 5: a = 3
        if q == 8: a = 1.5
        plt.axvline(x=max(i), color="gray", linestyle="--")
        plt.scatter(max(i), nClass[i.index(max(i))], color="red")
        plt.plot(i, nClass, label=NAMEMETRICS[q], linewidth=a)
        q +=1
    plt.legend(loc="best")
    plt.xlabel("Metric score")
    plt.ylabel("Num of class " + str(numClass))
    plt.title("Model" + str(numModel) + "Class" + str(numClass))
    plt.show()

def d3(namCl, w):
    #precision-racall-крива
    precision, recall, _ = skm.precision_recall_curve(df['GT'], namCl)
    maxEl = np.array((2 * precision * recall) / (precision + recall)).argmax()
    fig, ax = plt.subplots(1, 1)
    ax.plot(recall, precision)
    ax.plot(recall[maxEl], precision[maxEl], 'o', color='red')
    plt.xlabel('Recall')
    plt.ylabel("Precision")
    plt.title('PR curve for Model' + str(w))
    plt.show()

def d3_2(namCl, w):
    #ROC
    fpr1, tpr1, thres1 = roc_curve(df["GT"], namCl)
    youdens_j1 = tpr1 - fpr1
    plt.plot(fpr1, tpr1, label = 'ROC curve')
    plt.scatter(fpr1[np.argmax(youdens_j1)], tpr1[np.argmax(youdens_j1)], color='red')
    plt.title("ROC curve for Model" + str(w))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def main():
    print('Значення порогів:', thresholds)
    for i in thresholds:
        df["Model_1_" + str(i)] = [0 if j >= i else 1 for j in df["Model_1_0"]]
        df["Model_2_" + str(i)] = [1 if j >= i else 0 for j in df["Model_2_1"]]

    #обчислення значень метри для 1 моделі
    am1, pm1, rm1, fm1, mccm1, bam1, yjm1, aucprm1, aucrom1 = calcMetrics("Model_1_", 1)
    #обчислення значень метри для 2 моделі
    am2, pm2, rm2, fm2, mccm2, bam2, yjm2, aucprm2, aucrom2 = calcMetrics("Model_2_", 2)
    print('\n------TASK 3b-------')
    s = [am1, pm1, rm1, fm1, mccm1, bam1, yjm1, aucprm1, aucrom1, am2, pm2, rm2, fm2, mccm2, bam2, yjm2, aucprm2, aucrom2]
    plot3B(s)
    print('\n\n------TASK 3c-------')
    c3(1, 0, s)
    c3(2, 1, s)
    print('\n\n------TASK 3d-------')
    d3(1 - df['Model_1_0'],1)
    d3(df['Model_2_1'],2)
    d3_2(1 - df["Model_1_0"], 1)
    d3_2(df["Model_2_1"], 2)


main()
print("""Task4 
   ROC і PR криві є трішки кращими у першій моделі,ніж у другій.\n
    Припускаючи  перша модель є кращою """)
print('\n------TASK 5/6-------')
df = pd.read_csv("D:\Downloads\KM-03-3.csv")
db = '08-10'
K = int(db.split('-')[1]) % 4
percdel = 50 + 10 * K
cl1 = df[df['GT'] == 1]
print('\nВідсоток видалених об’єктів класу 1 -', percdel, '%')
countDel = (len(cl1) * percdel)/100
indices_to_remove = random.sample(cl1.index.tolist(), int(countDel))
df = df.drop(indices_to_remove)
print('Кількість елементів кожного класу після видалення\n', df['GT'].value_counts())

print('\n------TASK 7-------')
main()

print(""" TASK 8 \n
Спираючись на попередні результати можна сказати, що після видаленн ключові графіки не дуже сильно змінились. \n
Тому гадаю, що  1-ша модель видає кращий результат """)

print("""Task 9 \n
Незбалансована вибірка особливо не впливає на значення метрик. Модель 1 дала все ж дала трішкі кращі результати за модель 2\n
Незбаланосваність вибірки скоріше підкреслила трішки гіршу можливість 2 моделі виділяти класи у порівнянні з моделю 1.
У збалансованій вибірці ми отримали більш точні значень метрик(бо у не збалансованій помічаємо,що не відображає реальної ефективності алгоритму..різниці все ж не сильна)\n
отже збалансовані дані дозволяють отримати більш точні та репрезентативні результати класів, тому працювати з ними краще""")