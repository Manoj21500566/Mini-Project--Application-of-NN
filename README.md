# Mini-Project--Application-of-NN

## Project Title:
### Stock market prediction
## Project Description 
   We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.
## Algorithm:
1.import the necessary pakages.
2.install the csv file
3.using the for loop and predict the output
4.plot the graph
5.analyze the regression bar plot

## Program:
###                        import the necessary pakages
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
~~~
###                       install the csv file
~~~
df = pd.read_csv('/content/Tesla.csv')
df.head()
df.shape
df.describe()
df.info()
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)
df.isnull().sum()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
features = df[['open-close', 'low-high']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

print(f'{models[i]} : ')
print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
print()
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
~~~

## Output:

![1](https://user-images.githubusercontent.com/94588708/206670363-35c05133-31a9-4754-9083-9d71981cabc7.png)

![2](https://user-images.githubusercontent.com/94588708/206670381-852506c1-dc85-4de2-80a6-3bf958a4ee42.png)

![3](https://user-images.githubusercontent.com/94588708/206670396-e101a966-4e36-49f4-9a74-06e92c006435.png)

![4](https://user-images.githubusercontent.com/94588708/206670413-06c10e3f-99ad-401d-ac12-651c63c86471.png)

![5](https://user-images.githubusercontent.com/94588708/206670431-551c2bca-5d17-4a3c-b18f-c1509c3c36f7.png)

![6](https://user-images.githubusercontent.com/94588708/206670488-952ce015-68a3-4bef-88f7-1343278e0800.png)

![7](https://user-images.githubusercontent.com/94588708/206670512-cd72c675-5e62-4cc7-86d6-214d7d1ffa90.png)

![8](https://user-images.githubusercontent.com/94588708/206670573-1bc1387d-7b73-4e22-8234-b7b8a4a3f073.png)

![9](https://user-images.githubusercontent.com/94588708/206670624-43d0bd32-488f-4a42-bbca-1bb367d023d4.png)

![10](https://user-images.githubusercontent.com/94588708/206670649-152899db-5ead-4340-9472-5852ca5b7525.png)

![11](https://user-images.githubusercontent.com/94588708/206670695-79c5701a-e4a7-4b4f-8ec1-900f0c55e1a3.png)

![12](https://user-images.githubusercontent.com/94588708/206670716-58e0cc81-edbc-4940-a4e9-634ff7990e67.png)

![13](https://user-images.githubusercontent.com/94588708/206670747-2e9f820f-2cb3-41dc-b598-d91292306f53.png)

## Advantage :
Python is the most popular programming language in finance. Because it is an object-oriented and open-source language, it is used by many large corporations, including Google, for a variety of projects. Python can be used to import financial data such as stock quotes using the Pandas framework.
## Result:
Thus, stock market prediction is implemented successfully.

### Project by
Manoj M-212221240027
