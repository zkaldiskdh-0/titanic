# titanic
## 2019100755 정진영입니다.

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

train = pd.read_csv('./train.csv')

test = pd.read_csv('./test.csv')

train.head()

분석에 필요한 패키지들을 Import하고 데이터를 불러온다.

데이터 분석
---------
train.shape
(891, 12)

test.shape
(418, 11)

train.info() #데이터 정보분석

train.isnull().sum() #결측치 개수 정보

test.isnull().sum() 

데이터 시각화
------
def bar_chart(feature):
    
    # 각 column(=feature)에서 생존자 수 count
    
    survived = train[train['Survived']==1][feature].value_counts()
    
    # 각 column(=feature)에서 사망자 수 count
    dead = train[train['Survived']==0][feature].value_counts()
    
    # 생존자수, 사망자수를 하나의 dataframe으로 묶는다.
    df = pd.DataFrame([survived, dead])
    
    # 묶은 dataframe의 인덱스명(행 이름)을 지정한다.
    df.index = ['Survived', 'Dead']
    
    # plot을 그린다.
    
    df.plot(kind='bar', stacked=True, figsize=(10,5))

     #return df

bar_chart('Sex') #그래프로 시각화

bar_chart('Pclass')

bar_chart('Embarked')

bar_chart('SibSp')

데이터 시각화 후 분석결과 남<여 생존, 1등급, 가족존재 승객의 생존율이 높음, 탑승지역은 비율상 S 승객이 많이 사망

데이터 전처리
-----
### 데이터 전처리(성별)

train_test_data = [train, test]


sex_mapping = {"male":0, "female":1}

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

데이터 전처리(가족,동승자 여부)

for dataset in train_test_data:
   
    # 가족수 = 형제자매 + 부모님 + 자녀 + 본인
   
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
   
    dataset['IsAlone'] = 1
    
    
    # 가족수 > 1이면 동승자 있음
    
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0

### 데이터 전처리(승객 등급)

class_list=[]

for i in range(1,4):
   
    series = train[train['Pclass'] == i]['Embarked'].value_counts()
   
    class_list.append(series)


df = pd.DataFrame(class_list)

df.index = ['1st', '2nd', '3rd']

df.plot(kind="bar", figsize=(10,5))

결과 Q지역이 다른지역보다 등급이 낮아보이나, 승객 대부분이 S지역에서 탑승했으므로 Embarked의 결측치는 S로 채움

for dataset in train_test_data:
    
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {'S':0, 'C':1, 'Q':2}

for dataset in train_test_data:
    
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    

train.head()

### 데이터 전처리(이름)
이름으로 Mr,Mrs/ Ms로 혼인 여부를 판단 또 성별 혼인여부를 반영할 수 있으므로 추출

for dataset in train_test_data:
    
    dataset['Title'] = dataset['Name'].str.extract('([\w]+)\.', expand=False)

train.head()#확인

train['Title'].value_counts()

test['Title'].value_counts()

for dataset in train_test_data:
   
    dataset['Title'] = dataset['Title'].apply(lambda x: 0 if x=="Mr" else 1 if x=="Miss" else 2 if x=="Mrs" else 3 if x=="Master" else 4) #각 호칭들을 숫자에 매핑, Mr Miss Mrs 이외는 하나로 취급

train['Title'].value_counts()

test['Title'].value_counts()

bar_chart('Title') # 분포 확인후 시각화

시각화 결과 Mr의 사망률이 높음, 또 혼인하지않은 Miss가 사망률이 더 높다. Master의 경우 유아가 많아 생존률이 더 높음

### 데이터 전처리(방 번호)

train['Cabin'].value_counts()

train['Cabin'] = train['Cabin'].str[:1] # 숫자제외 알파벳 추출

class_list=[]

for i in range(1,4):
    
    a = train[train['Pclass'] == i]['Cabin'].value_counts()
    
    class_list.append(a)


df = pd.DataFrame(class_list)

df.index = ['1st', '2nd', '3rd']

df.plot(kind="bar", figsize=(10,5))

등급별 생존율로 보았을 때 1등급이 가장 높고 3등급의 사망률이 가장 높다.

### 데이터 전처리(나이)

for dataset in train_test_data:
    dataset['Age'].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True #그룹으로 묶고 그룹별 중간값으로 결측치를 대체

g = sns.FacetGrid(train, hue="Survived", aspect=4)
g = (g.map(sns.kdeplot, "Age").add_legend()) # add_legend()는 범주를 추가하는 파라미터이다.

for dataset in train_test_data:
    dataset['Agebin'] = pd.cut(dataset['Age'], 5, labels=[0,1,2,3,4]) #그룹을 청소년,청년,중년,장년,노년의 5개로 나눔

### 데이터 전처리(요금)

for dataset in train_test_data:
    
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True) # 요금은 등급이 높을수록 비싸니, 등급별 중간값으로 결측치 대체

g = sns.FacetGrid(train, hue="Survived", aspect=4)

g = (g.map(sns.kdeplot, "Fare")
     .add_legend() # 범주 추가
     .set(xlim=(0, train['Fare'].max()))) # x축 범위 설정

for dataset in train_test_data:
   
    dataset['Farebin'] = pd.qcut(dataset['Fare'], 4, labels=[0,1,2,3])

pd.qcut(train['Fare'], 4) #확인

drop_column = ['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']

for dataset in train_test_data:
    
    dataset = dataset.drop(drop_column, axis=1, inplace=True) #전처리 이후 훈련에 사용되지않는 Column삭제

train.head()#확인

train.info()

test.info()

데이터학습
---------
drop_column2 = ['PassengerId', 'Survived']

train_data = train.drop(drop_column2, axis=1)

target = train['Survived'] #PassengerId는 이용하지 없는 데이터이므로 삭제, Survived는 결과값에 해당하므로 삭제

from sklearn.tree import DecisionTreeClassifier # 의사결정나무

from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트

from sklearn.naive_bayes import GaussianNB # 나이브 베이즈 분류

from sklearn.svm import SVC # 서포트 벡터 머신

from sklearn.linear_model import LogisticRegression # 로지스틱 회귀

clf = LogisticRegression()

clf.fit(train_data, target)

clf.score(train_data, target) #모델별 점수 출력 (로지스틱 회귀)

clf = DecisionTreeClassifier()

clf.fit(train_data, target)

clf.score(train_data, target) #모델별 점수 출력 (의사결정나무)

clf = RandomForestClassifier()

clf.fit(train_data, target)

clf.score(train_data, target) #모델별 점수 출력 (랜덤결정나무)

clf = GaussianNB()

clf.fit(train_data, target)

clf.score(train_data, target) #모델별 점수 출력 (나이브 베이즈 분류)

clf = SVC()

clf.fit(train_data, target)

clf.score(train_data, target) #모델별 점수 출력 (서포트 벡터 머신)

clf = DecisionTreeClassifier()

clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1)

predict = clf.predict(test_data) 

submission = pd.DataFrame({
    
    'PassengerId' : test['PassengerId'],
    
    'Survived' : predict})


submission.to_csv('submission.csv', index=False) # 가장 점수가 높은 의사결정나무를 적용 후 PassengerId삭제

submission = pd.read_csv("submission.csv")

submission.head() #예측결과를 PassengerId와 매치시켜 데이터프레임으로 묶은뒤 저장
