import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

telco_data = pd.read_csv('data/telco.csv')

# ================================ 데이터 전처리 ================================

telco_data['성별']  = telco_data['성별'].replace('남',0)
telco_data['성별']  = telco_data['성별'].replace('여',1)
telco_data['요금제'] = telco_data['요금제'].replace('CAT 50',1)
telco_data['요금제'] = telco_data['요금제'].replace('CAT 100',2)
telco_data['요금제'] = telco_data['요금제'].replace('CAT 200',3)
telco_data['요금제'] = telco_data['요금제'].replace('Play 100',4)
telco_data['요금제'] = telco_data['요금제'].replace('Play 300',5)

telco_data['지불방법'] = telco_data['지불방법'].replace('선불',0)
telco_data['지불방법'] = telco_data['지불방법'].replace('후불',1)

telco_data['이탈여부'] = telco_data['이탈여부'].replace('이탈',0)
telco_data['이탈여부'] = telco_data['이탈여부'].replace('유지',1)

telco_data['통화량구분'] = telco_data['통화량구분'].replace('무',1)
telco_data['통화량구분'] = telco_data['통화량구분'].replace('저',2)
telco_data['통화량구분'] = telco_data['통화량구분'].replace('중저',3)
telco_data['통화량구분'] = telco_data['통화량구분'].replace('중',4)
telco_data['통화량구분'] = telco_data['통화량구분'].replace('중고',5)
telco_data['통화량구분'] = telco_data['통화량구분'].replace('고',6)

telco_data['납부여부'] = telco_data['납부여부'].replace('High CAT 50',1)
telco_data['납부여부'] = telco_data['납부여부'].replace('High CAT 100',2)
telco_data['납부여부'] = telco_data['납부여부'].replace('High Play 100',3)
telco_data['납부여부'] = telco_data['납부여부'].replace('OK',4)

telco_data['통화품질불만'] = telco_data['통화품질불만'].replace('F',0)
telco_data['통화품질불만'] = telco_data['통화품질불만'].replace('T',1)

telco_data['미사용'] = telco_data['미사용'].replace('F',0)
telco_data['미사용'] = telco_data['미사용'].replace('T',1)

Y = np.array(pd.DataFrame(telco_data, columns=['이탈여부']))

X = np.array(pd.DataFrame(telco_data, columns=['성별','연령','서비스기간','단선횟수','지불방법','요금제',
'주간통화시간_분','야간통화시간_분','주말통화시간_분','국제통화시간_분','국내통화시간_분','총통화시간_분',
'통화량구분','납부여부','평균납부요금','통화품질불만','미사용']))

feature_names = ['성별','연령','서비스기간','단선횟수','지불방법','요금제',
'주간통화시간_분','야간통화시간_분','주말통화시간_분','국제통화시간_분','국내통화시간_분','총통화시간_분',
'통화량구분','납부여부','평균납부요금','통화품질불만','미사용']

# feature_names = ['성별','연령','서비스기간','단선횟수','지불방법',
# '요금제','통화량구분','총통화요금','부과요금','납부여부','평균납부요금','주간통화비율',
# '야간통화비율','주말통화비율','국제통화비율','통화품질불만','미사용']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)


print('Y train : ')
print(Y_train)
print('\nY test : ')
print(Y_test)
print('\n')
print('X train: ')
print(X_train[0])
print('\nX test : ')
print(X_test)

# ================================ DecisionTreeClassifier ================================

criterion= 'entropy'
min_impurity_split = 0.1

clf = tree.DecisionTreeClassifier(criterion=criterion, min_impurity_split=min_impurity_split, max_depth=5)
# clf = tree.DecisionTreeClassifier(max_depth=7)
clf = clf.fit(X_train, Y_train)

pred = clf.predict(X_test)

prec = precision_score(Y_test, pred, average='macro')
rec = recall_score(Y_test,pred, average='macro')
f1 = f1_score(Y_test, pred ,average='macro')
acc = accuracy_score(Y_test,pred)

print("\n")
print("criterion: {}".format(criterion))
print("impurity split threshold {}".format(min_impurity_split))
print("precision_score: {:.2f} , ".format(prec),prec)
print("recall_score: {:.2f} , ".format(rec), rec)
print("F1_score: {:.2f}, ".format(f1), f1)
print("Acc : {:.2f}, ".format(acc), acc)
print("\n")

print("Feature importances\n")
a=0
for i in feature_names:
    print(i)
    print(clf.feature_importances_[a])
    a +=1

#feature importance 시각화
n_feature = X.shape[1]
idx = np.arange(n_feature)
font_name = font_manager.FontProperties(fname='C:/NanumSquare_acB.ttf').get_name()
rc('font', family=font_name)
plt.barh(idx, clf.feature_importances_, align='center')
plt.yticks(idx, feature_names, fontsize=6)
plt.xlabel('feature importance', size=10)
plt.ylabel('feature', size=10)
plt.show()
