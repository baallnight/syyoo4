import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ctx = '../data/'
train = pd.read_csv(ctx + 'train.csv')
test = pd.read_csv(ctx + 'test.csv')
# df = pd.DataFrame(train)
# print(df.columns)
"""
['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

PassengerId 고객아이디      
Survived 생존여부     Survival    0 = No, 1 = Yes
Pclass 승선권 클래스    Ticket class    1 = 1st, 2 = 2nd, 3 = 3rd
Name 이름
Sex  성별  Sex    
Age  나이  Age in years    
SibSp  동반한 형제자매, 배우자 수  # of siblings / spouses aboard the Titanic    
Parch  동반한 부모, 자식 수  # of parents / children aboard the Titanic    
Ticket  티켓 번호  Ticket number    
Fare  티켓의 요금  Passenger fare    
Cabin  객실번호  Cabin number    
Embarked  승선한 항구명  Port of Embarkation  
  C = Cherbourg 쉐부로, Q = Queenstown 퀸스타운, S = Southampton 사우스햄톤

"""
f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Survived'].value_counts().plot.pie(explode=[0, 0.1],
                                          autopct="%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')

sns.countplot('Survived', data=train, ax=ax[1])
ax[1].set_title('Survived')
# plt.show() 생존률 38.4% 사망률 61.6%

"""
데이터는 훈련데이터(train.csv) , 목적데이터(test.csv) 두가지로
제공됩니다. 
목적데이터에는 위 항목에서는  Survived 정보가 빠져있습니다.
그것은 답이기 때문입니다.
"""
# *************
# 성별
# *************

f, ax = plt.subplots(1, 2, figsize=(18, 8))
train['Survived'][train["Sex"] == 'male'].value_counts().plot.pie(explode=[0, 0.1],
                                                                  autopct="%1.1f%%", ax=ax[0], shadow=True)
train['Survived'][train["Sex"] == 'female'].value_counts().plot.pie(explode=[0, 0.1],
                                                                    autopct="%1.1f%%", ax=ax[1], shadow=True)
ax[0].set_title('Survived(Male)')
ax[1].set_title('Survived(Female)')

# plt.show()
# 남성 생존률 18.9% 사망률 81.1%
# 여성 생존률 74.2% 사망률 25.8%

# *************
# 승선권 Pclass
# *************

df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']
df = pd.crosstab(df_1, df_2, margins=True)
# print(df.head())
"""
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
"""
# Embarked 는 배를 탄 항구
f, ax = plt.subplots(2, 2, figsize=(20, 15))
sns.countplot('Embarked', data=train, ax=ax[0, 0])
ax[0, 0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked', hue='Sex', data=train, ax=ax[0, 1])
ax[0, 1].set_title('Male - Female for Embarked')
sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1, 0])
ax[1, 0].set_title('Embarked vs Survived')
sns.countplot('Pclass', data=train, ax=ax[1, 1])
ax[1, 1].set_title('Embarked vs PClass')

# plt.show()
"""
위 데이터를 보면 절반 이상의 승객이 ‘Southampton’에서 배를 탔으며,
 여기에서 탑승한 승객의 70% 가량이 남성이었습니다.
 현재까지 검토한 내용으로는 남성의 사망률이 여성보다 
 훨씬 높았기에 자연스럽게 ‘Southampton’에서 
 탑승한 승객의 사망률이 높게 나왔습니다.
또한 ‘Cherbourg’에서 탑승한 승객들은 
1등 객실 승객의 비중 및 생존률이 높은 것으로 보아서 
이 동네는 부자동네라는 것을 예상할 수 있습니다.
"""

# 결측치 제거

# train.info()
"""
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
"""
#print(train.isnull().sum())
"""
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
"""


def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['survived', 'dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.show()


#bar_chart('Sex')
#bar_chart('Pclass')
#bar_chart('SibSp')
#bar_chart('Parch')
#bar_chart('Embarked')

# Cabin, Ticket 값 삭제

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

# Embark 값 가공
s_city = train[train['Embarked'] == 'S'].shape[0] # 0차원은 값이 1개인 상태
c_city = train[train['Embarked'] == 'C'].shape[0]
q_city = train[train['Embarked'] == 'Q'].shape[0]

#print("S = {}, C = {}, Q = {}".format(s_city, c_city, q_city))
#print("S : ", s_city) # 644
#print("C : ", c_city) # 168
#print("Q : ", q_city) # 77

train = train.fillna({"Embarked":"S"}) # null 값을 채운다
city_mapping = {"S": 1, "C": 2, "Q": 3}
train["Embarked"] = train["Embarked"].map(city_mapping)
test["Embarked"] = test["Embarked"].map(city_mapping)

#print(train.head())
#print(test.head())

combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])

#print(pd.crosstab(train['Title'], train['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

"""
    Title  Survived
0  Master  0.575000
1    Miss  0.702703
2      Mr  0.156673
3     Mrs  0.793651
4    Rare  0.285714
5   Royal  1.000000
"""

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0) # fillna
train.head()

train = train.drop(['Name','PassengerId'], axis = 1)
test = test.drop(['Name','PassengerId'], axis = 1)
combine = [train, test]


sex_mapping = {"male": 0, "female": 1}
train["Sex"] = train["Sex"].map(sex_mapping)
test["Sex"] = test["Sex"].map(sex_mapping)
#print(train.head())

#Age 가공하기
train["Age"] = train["Age"].fillna(-0.5)
train["Age"] = train["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)

#print(train.head())

age_title_mapping = {0: "Unknown", 1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train['AgeGroup'])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test['AgeGroup'])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
#print(train.head())

age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
#train.head()

train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1,2,3,4]) # 4등분한다.
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1,2,3,4]) # 4등분한다.

train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)


# ***
# 데이터 모델링
# ***
train_data = train.drop('Survived', axis = 1)
target = train['Survived']
#print( train_data.shape);
#print(target.shape);

#print(train.info)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train2, test2 = train_test_split(train, test_size=0.3, random_state=0)
target_col = ['Pclass', 'Sex', 'Embarked']
train_X = train2[target_col]
train_Y = train2['Survived']
test_X = test2[target_col]
test_Y = test2['Survived']

features_one = train_X.values
target = train_Y.values

tree_model = DecisionTreeClassifier()
tree_model.fit(features_one, target)
dt_prediction = tree_model.predict(test_X)
print('The accuracy of the Decision Tree is ',
      metrics.accuracy_score(dt_prediction, test_Y))

# The accuracy of the Decision Tree is  0.8097014925373134

# ***
# KNN 모델링
# ***
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# 10개의 split을 사용
clf = KNeighborsClassifier(n_neighbors=13)
scoring = 'accuracy'

print('----- train data -----')
print(train_data)
print('----- target -----')
print(target)
score = cross_val_score(clf, train_data, target, cv=k_fold,
                        n_jobs=1, scoring='accuracy')

print("스코어: {}".format(score))
# knn 스코어
print(np.mean(score) * 100, 2)

prediction = clf.predict(test_)