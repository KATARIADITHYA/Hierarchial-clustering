# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:32:46 2022

@author: KATARI ADITHYA
"""


import numpy as np
import pandas as pd
df=pd.read_csv('D:/Desktop/datasets1/crime_data.csv',encoding= 'unicode_escape')
df.columns

#datapreprocessing
df.duplicated()
df.drop_duplicates(inplace=True)
df.describe()
for i in df.iloc[:,1:].columns:
    IQR=(df[i].quantile(0.75))-(df[i].quantile(0.25))
    ll=df[i].quantile(0.25)-(1.5*IQR)
    ul=df[i].quantile(0.75)+(1.5*IQR)
    df[i]=np.where(df[i]>ul,ul,np.where(df[i]<ll,ll,df[i]))
l=[]
for i in df.iloc[:,1:].columns:
    l.append(df[i].var()==0)
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
d1=pd.DataFrame(enc.fit_transform(df.iloc[:,[0]]).toarray())
for i in d1.columns:
    df[i]=d1[i]
df=df.iloc[:,1:]
df.isna().sum()
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm(df.iloc[:,:])

#EDA
#Measures of Central Tendency
df_norm.mean()
df_norm.median()
df_norm.mode()

# Measures of Dispersion
df_norm.var()
df_norm.std()
r=df_norm.min()-df_norm.max()
r

#Measure of asymmetry
df_norm.skew()

#Measure of peakedness
df_norm.kurt()

#Graphical  representation
import matplotlib.pyplot as plt
import numpy as np
df_norm.columns
#histogram
plt.hist(df['Murder'])
plt.hist(df['Assault'])
plt.hist(df['UrbanPop'])
plt.hist(df['Rape'])

#boxplot
plt.boxplot(df['Murder'])
plt.boxplot(df['Assault'])
plt.boxplot(df['UrbanPop'])
plt.boxplot(df['Rape'])

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
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean').fit(df_norm)
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df_norm['clust'] = cluster_labels
dfnorm1=df_norm.iloc[:,[54,0,1,2,3]]
dfnorm1.head(10)
dfnorm1.iloc[:,1:5].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,5:].groupby(dfnorm1.clust).mean()
dfnorm1.to_csv("heirarchial2.csv", encoding = "utf-8")

'''Crime rate clust 1>clust 2>clust 0'''










