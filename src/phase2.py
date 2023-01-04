import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import csv
from linear_model import LeastSquaresBias

def euclidean_dist_squared(X, Xtest):
    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


filename = "phase2_training_data.csv"
with open(os.path.join("..","data",filename),"rb") as f:
    df0 = pd.read_csv(f)

df0.head()

#reorganize the dataset
df = df0.pivot_table(index="date",columns='country_id',values=['deaths','cases','cases_14_100k','cases_100k'])
dates = [dt.datetime.strptime(date, "%m/%d/%Y").date() for date in df.index.values]
df = df.iloc[np.argsort(dates),:]
df.head()

#extract death information
df_deaths = df['deaths']
df_deaths.head()
# daily deaths
df_diff0 = df_deaths.diff(axis=0)


df_diff=df_diff0.iloc[180:280,:]
print(df_diff.shape)
euclid_dis = euclidean_dist_squared(np.array(df_diff['CA'])[None], np.array(df_diff).T)
# sorted countries close to Canada in terms of daily deaths
df_diff.columns.values[np.argsort(euclid_dis.flatten())[range(10)]]


#compute the lag of daily death of canada
daily_death_ca=df_diff0['CA']
daily_death_ca_lag1=daily_death_ca.shift(periods=1)
daily_death_ca_lag2=daily_death_ca.shift(periods=2)
daily_death_ca_lag3=daily_death_ca.shift(periods=3)

feature_space=pd.concat([daily_death_ca,daily_death_ca_lag1,daily_death_ca_lag2,daily_death_ca_lag3],axis=1)
feature_space.columns=["daily_death_ca","daily_death_ca_lag1","daily_death_ca_lag2","daily_death_ca_lag3"]
fs_sub=feature_space.iloc[200:300,:]

print(fs_sub.head())

model=LeastSquaresBias()
X=feature_space.iloc[200:300,1:4]
y=feature_space.iloc[200:300,0]
model.fit(X=X,y=y)
print(model.w)

#3.08624767  0.46034715 -0.12275157  0.38948886

dat_pred = feature_space
for i in range(5):
    new_data = np.array([dat_pred.iloc[-1,0], dat_pred.iloc[-2,0], dat_pred.iloc[-3,0]])[None]
    print(new_data)
    y_pred = model.predict(X=new_data)
    dat_pred = pd.concat([dat_pred, pd.DataFrame(np.append(y_pred, new_data[0])[None], columns=dat_pred.columns.values)], axis=0)

pred_deaths_CA = np.cumsum(dat_pred.iloc[1:,0])

#compute the lag of daily death of canada
death_ca=df_deaths['CA']
print(death_ca)
death_ca_lag1=death_ca.shift(periods=1)
death_ca_lag2=death_ca.shift(periods=2)
death_ca_lag3=death_ca.shift(periods=3)

feature_space=pd.concat([death_ca,death_ca_lag1,death_ca_lag2,death_ca_lag3],axis=1)
feature_space.columns=["death_ca","death_ca_lag1","death_ca_lag2","death_ca_lag3"]

model=LeastSquaresBias()
X=feature_space.iloc[200:300,1:4]
y=feature_space.iloc[200:300,0]
model.fit(X=X,y=y)

dat_pred = feature_space
for i in range(5):
    new_data = np.array([dat_pred.iloc[-1,0], dat_pred.iloc[-2,0], dat_pred.iloc[-3,0]])[None]
    y_pred = model.predict(X=new_data)
    dat_pred = pd.concat([dat_pred, pd.DataFrame(np.append(y_pred, new_data[0])[None], columns=dat_pred.columns.values)], axis=0)

pred_deaths_CA2 = dat_pred.iloc[:,0]

#write the prediction results
prediction = (pred_deaths_CA2[-5:]+pred_deaths_CA[-5:])/2
prediction.to_csv("../data/phase2_prediction.csv", index = False, sep = ",")