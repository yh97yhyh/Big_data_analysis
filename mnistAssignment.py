import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def get_one_fold(data, turn, fold=10):
    tot_length = len(data)
    each = int(tot_length/fold)
    mask = np.array([True if each*turn <= i < each*(turn+1)
                     else False
                     for i in list(range(tot_length))])
    return data[~mask], data[mask]

def runCV(clf, shuffled_data, shuffled_labels, fold=10, isAcc=True):
    from sklearn.metrics import precision_recall_fscore_support
    results = []
    for i in range(fold):
        train_data, test_data = get_one_fold(shuffled_data, i, fold=fold)
        train_labels, test_labels = get_one_fold(shuffled_labels, i, fold=fold)
        clf = clf.fit(train_data, train_labels)
        pred = clf.predict(test_data)
        correct = pred==test_labels
        if isAcc:
            acc = sum([1 if x == True else 0 for x in correct])/len(correct)
            results.append(acc)
        else:
            results.append(precision_recall_fscore_support(pred, test_labels))
    return results


# =============== 데이터 전처리 ===============

f = open('data/train_mnist.csv', 'r')

f.readline()

digits = [] #이미지의 각 행의 픽셀값 (데이터값)
digit_labels = [] #이미지의 각 행의 라벨값

for line in f.readlines():
    splitted = line.replace("\n", "").split(",")
    digit_labels.append(int(splitted[0]))
    digits.append(np.array(splitted[1:], dtype = np.float32))

digits = np.array(digits)
norm_digits = digits/255; # 정규화
digit_labels = np.array(digit_labels)
f.close()

# =============== 1. DecisionTreeClassifier ===============

np.random.seed(0)
numbers = list(range(len(norm_digits)))
np.random.shuffle(numbers)
shuffled_data = norm_digits[numbers]
shuffled_labels = digit_labels[numbers]

print("=============== 1. DecisionTreeClassifier ===============")
clf = DecisionTreeClassifier()
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')


# =============== 2. GaussianNB ===============

print("=============== 2. GaussianNB ===============")
clf = GaussianNB()
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')

# =============== 3. KNeighborsClassifier ===============

print("=============== 3. KNeighborsClassifier ===============")
clf = KNeighborsClassifier()
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')

# =============== 4. LogisticRegression ===============

print("=============== 4. LogisticRegression ===============")
clf = LogisticRegression()
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')

# =============== 5. Perceptron ===============

print("=============== 5. Perceptron ===============")
clf = Perceptron(max_iter=500, n_jobs=3)
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')

#  =============== 6. MLPClassifier ===============

print("=============== 6. MLPClassifier ===============")
clf = MLPClassifier(hidden_layer_sizes=20, max_iter=500)
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')

# =============== 7. RandomForestClassifier ===============

print("=============== 7. RandomForestClassifier ===============")
clf = RandomForestClassifier()
results = runCV(clf, shuffled_data, shuffled_labels, isAcc=True)
acc = sum(results) / 10
print(results)
print(acc)
print('\n')
