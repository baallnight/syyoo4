from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

np.random.seed(0)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
print(df.columns)
"""
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']
"""

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75 # 25%
train, test = df[df['is_train']==True], df[df['is_train']==False]
features = df.columns[:4] # 앞에서 4번째 컬럼까지 취해라
y = pd.factorize(train['species'])[0]

# ***
# 러닝
# ***

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y)
print(clf.predict(test[features]))

# ***
# 테스팅 (정확도 평가)
# ***

preds = iris.target_names[clf.predict(test[features])]
print("--- 크로스 탭 결과---")

print(pd.crosstab(test['species'], preds, rownames=['Actural Species'],
                  colnames=['Predicated Species']))
print('피쳐별 중요도')
print(list(zip(train[features], clf.feature_importances_)))
