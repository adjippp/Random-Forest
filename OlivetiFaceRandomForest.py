from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
people = fetch_lfw_people(min_faces_per_person = 70, resize=.4)
print(dir(people))

df=pd.DataFrame(people['data'])
df['target']=people['target']
df['nama']=df['target'].apply(
    lambda x: people['target_names'][x]
)

x=df.iloc[:,:-2]
y=df.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1)
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.predict(x_test))
print(y_test)
