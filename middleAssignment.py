import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score, recall_score, f1_score
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import pydotplus as pydotplus
import seaborn as sns
import os
from pylab import rcParams
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# =============================== 데이터 전처리 ===============================

telco_data = pd.read_csv('data/telco.csv')

# to numeric
telco_data['성별']  = telco_data['성별'].replace('남',0)
telco_data['성별']  = telco_data['성별'].replace('여',1)
telco_data['요금제'] = telco_data['요금제'].replace('CAT 50',1)
telco_data['요금제'] = telco_data['요금제'].replace('CAT 100',2)
telco_data['요금제'] = telco_data['요금제'].replace('CAT 200',3)
telco_data['요금제'] = telco_data['요금제'].replace('Play 100',4)
telco_data['요금제'] = telco_data['요금제'].replace('Play 300',5)
telco_data['이탈여부'] = telco_data['이탈여부'].replace('이탈',1)
telco_data['이탈여부'] = telco_data['이탈여부'].replace('유지',0)
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

tmp = []
for each in telco_data['국제통화시간_분']: # 0 ~ 327
    if each <= 50:
        tmp.append(1)
    elif each <= 100:
        tmp.append(2)
    elif each <= 150:
        tmp.append(3)
    elif each <= 200:
        tmp.append(4)
    elif each <= 250:
        tmp.append(5)
    elif each <= 300:
        tmp.append(6)
    else:
        tmp.append(7)
telco_data['국제통화시간_분'] = tmp

tmp = []
for each in telco_data['연령']: # 12 ~ 80
    if each <= 15:
        tmp.append(1)
    elif each <= 20:
        tmp.append(2)
    elif each <= 25:
        tmp.append(3)
    elif each <= 30:
        tmp.append(4)
    elif each <= 35:
        tmp.append(5)
    elif each <= 40:
        tmp.append(6)
    elif each <= 45:
        tmp.append(7)
    elif each <= 50:
        tmp.append(8)
    elif each <= 55:
        tmp.append(9)
    elif each <= 60:
        tmp.append(10)
    elif each <= 65:
        tmp.append(11)
    elif each <= 70:
        tmp.append(12)
    elif each <= 75:
        tmp.append(13)
    else:
        tmp.append(14)
telco_data['연령'] = tmp

# data set
Y = np.array(pd.DataFrame(telco_data, columns=['이탈여부']))
X = np.array(pd.DataFrame(telco_data, columns=['성별','연령','서비스기간',
'요금제', '국제통화시간_분', '통화량구분',
'납부여부','평균납부요금','통화품질불만', '미사용']))
feature_names = ['성별','연령','서비스기간','요금제','국제통화시간_분',
'통화량구분','납부여부','평균납부요금','통화품질불만', '미사용']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# =============================== 모델 튜닝 ===============================

train_result = []
test_result = []
model_criterion = []
model_max_depth = []

insert_criterion = ['gini','entropy']
list_max_depth = [4, 8, 12, 15, 20] # depth가 깊어질수록 train score는 상승 하나 test score는 하락한다. 과적합 되었기 때문
for i in insert_criterion:
    for n in list_max_depth:
            DesTree = tree.DecisionTreeClassifier(criterion=i, max_depth=n)
            DesTree.fit(X_train,Y_train)
            train_result.append(DesTree.score(X_train,Y_train))
            test_result.append(DesTree.score(X_test,Y_test))
            model_criterion.append(i)
            model_max_depth.append(n)

result = pd.DataFrame()
result["Criterion"] = model_criterion
result["Depth"] = model_max_depth
result["TrainAccuracy"] = train_result
result["TestAccuracy"] = test_result
print("\n -- 모델 튜닝 결과 -- ")
print(result)


sorted_result = result.sort_values(by='TestAccuracy', ascending=False)

print("\n -- 최적 모델 -- ")
print(sorted_result.head(1))

criterion = sorted_result.head(1)['Criterion'].iloc[0]
max_depth = sorted_result.head(1)["Depth"].iloc[0]

clf = tree.DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
clf.fit(X_train, Y_train)

pred = clf.predict(X_test)
prec = precision_score(Y_test, pred, average='macro')
rec = recall_score(Y_test,pred, average='macro')
f1 = f1_score(Y_test, pred ,average='macro')
acc = accuracy_score(Y_test,pred)

# =============================== 결과 출력 ===============================

print("criterion : {}".format(criterion))
print("max depth : {}".format(max_depth))
print("precision_score : {:.2f} , ".format(prec),prec)
print("recall_score : {:.2f} , ".format(rec), rec)
print("F1_score : {:.2f}, ".format(f1), f1)
print("Acc : {:.2f}, ".format(acc), acc)
print("\n")

print(" -- Feature Importance --")
a=0
for i in feature_names:
    print(i)
    print(clf.feature_importances_[a])
    a+=1
    
# =============================== 시각화 ===============================

# feature importance 그래프 시각화
# 1. 통화품질불만, 2. 국제통화시간_분. 3. 연령, 4. 통화량구분 5. 납부여부
font_name = font_manager.FontProperties(fname='NanumSquare_acB.ttf').get_name()
rc('font', family=font_name)
n_feature = X.shape[1]
idx = np.arange(n_feature)
plt.barh(idx, clf.feature_importances_, align='center', color='salmon')
plt.yticks(idx, feature_names, fontsize=6)
plt.xlabel('feature importance', size=10)
plt.ylabel('feature', size=10)
plt.show()

# decision tree 시각화
# 한글로 저장할 수 있게 _export.py 파일의 fontname 'sans'로 변경
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=['유지','이탈'],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.encode('utf-8'))
graph.write_png('tree1.png')

# 중요한 attribute와 유지, 이탈의 상관관계 시각화

# 1. 통화품질불만
telco_data['통화품질불만'].unique()
sns.catplot(y="이탈여부", x="통화품질불만", data=telco_data, kind="bar", palette="Pastel1")
data = pd.get_dummies(data=telco_data, columns=['통화품질불만'])
plt.show()

# 2. 국제통화시간_분
sns.catplot(y="이탈여부", x="국제통화시간_분", data=telco_data, kind="bar", palette="Pastel1")
plt.show()

# 3. 연령
sns.catplot(y="이탈여부", x="연령", data=telco_data, kind="bar", palette="Pastel1")
plt.show()

# 4. 통화량구분
sns.catplot(y="이탈여부", x="통화량구분", data=telco_data, kind="bar", palette="Pastel1")
plt.show()

# 5. 납부여부
sns.catplot(y="이탈여부", x="납부여부", data=telco_data, kind="bar", palette="Pastel1")
plt.show()
