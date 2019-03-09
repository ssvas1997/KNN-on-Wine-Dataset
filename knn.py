import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("E:\Srinivas\KNN\wine.csv")
df.dtypes


features = df.drop('class',axis=1).values

classes = df['class'].values

(train_feat,test_feat,train_class,test_class) = train_test_split(features,classes,train_size = 0.8,random_state=40)

knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(train_feat,train_class)
pred = knn.predict(test_feat)


print("accuracy",metrics.accuracy_score(test_class,pred))
#df.describe()
dft = pd.read_csv("E:\Srinivas\KNN\wine2.csv")
feat = dft[dft.columns].values
pred = knn.predict(feat)
print("Target Class :",pred)

neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(train_feat,train_class)
    train_accuracy[i]= knn.score(train_feat,train_class)
    test_accuracy[i] = knn.score(test_feat,test_class)
    print("test",test_accuracy[i])
    print("train",train_accuracy[i])
    
plt.title("KNN Varying number of neighbors ")    
plt.plot(neighbors,test_accuracy,label='Testing Accuracy')
plt.plot(neighbors,train_accuracy,label = 'Training Accuracy')
plt.legend()
plt.xlabel("Number of Neighbor")
plt.ylabel('Accuracy')
plt.show()
