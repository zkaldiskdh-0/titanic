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
