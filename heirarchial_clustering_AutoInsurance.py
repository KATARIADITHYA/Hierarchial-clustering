# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:55:11 2022

@author: KATARI ADITHYA
"""

import numpy as np
import pandas as pd
df=pd.read_csv("D:/Desktop/datasets1/AutoInsurance.csv")
df.columns
df['Customer'].duplicated().sum()
#datapreprocessing
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.describe()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df3=df.iloc[:,[1,3,4,5,7,8,10,11,17,18,19,20,22]]
df = pd.get_dummies(df, columns = df3.columns)
df['Vehicle Size']= label_encoder.fit_transform(df['Vehicle Size'])
df.drop(df.columns[[0,2]],axis=1,inplace=True)
df.columns
for i in df.columns:
    IQR=(df[i].quantile(0.75))-(df[i].quantile(0.25))
    ll=df[i].quantile(0.25)-(1.5*IQR)
    ul=df[i].quantile(0.75)+(1.5*IQR)
    df[i]=np.where(df[i]>ul,ul,np.where(df[i]<ll,ll,df[i]))
l=[]
for i in df.columns:
    l.append(df[i].var()==0)
j=[]
for i in range(0,len(l)):
    if(l[i]==True):
        j.append(i)   
df.drop(df.columns[j],axis=1,inplace=True)
df.isna().sum()
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm(df.iloc[:,:])
#EDA
#Measures of Central Tendency
df_norm.mean()
df_norm.median()
# Measures of Dispersion
df_norm.var()
df_norm.std()
r=df_norm.min()-df_norm.max()
r

#Measure of asymmetry
df_norm.skew()

#Measure of peakedness
df_norm.kurt()
df.columns
#Graphical  representation
import matplotlib.pyplot as plt
import numpy as np
df_norm.columns
#histogram
import seaborn as sns
#Histogram
for i, p in enumerate(df_norm):
    plt.figure(i)
    sns.histplot(data=df_norm,x=p)
    

#Model building
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
z = linkage(df_norm, method = "complete", metric = "euclidean")
z
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,
    leaf_font_size = 10 
)
plt.show()
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = 'euclidean').fit(df_norm)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df_norm['clust'] = cluster_labels
df_norm.columns
dfnorm1=df_norm.iloc[:,[28,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]
dfnorm1.head(10)
dfnorm1.iloc[:,1:4].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,4:7].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,7:9].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,9:11].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,11:13].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,13:16].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,16:18].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,18:20].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,20:22].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,22:25].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,25:28].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,28:30].groupby(dfnorm1.clust).mean()


"cluster 2 has claimed more amount who are unemployed renew offer type offer 1"










