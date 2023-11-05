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
데이터 전처리(성별)

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

데이터 전처리(승객 등급)

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
