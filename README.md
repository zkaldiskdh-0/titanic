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

bar_chart('Sex')
bar_chart('Pclass')
bar_chart('Embarked')
bar_chart('SibSp')
