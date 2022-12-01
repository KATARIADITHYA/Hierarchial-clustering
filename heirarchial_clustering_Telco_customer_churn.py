# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:48:48 2022

@author: KATARI ADITHYA
"""

import numpy as np
import pandas as pd
df=pd.read_excel("D:/Desktop/datasets1/Telco_customer_churn.xlsx")
df.columns
df=pd.DataFrame(df,columns=['Number of Referrals','Tenure in Months','Avg Monthly Long Distance Charges','Avg Monthly GB Download','Monthly Charge','Total Charges','Total Refunds','Total Extra Data Charges','Total Long Distance Charges','Total Revenue'])

for i in df.columns:
    IQR=(df[i].quantile(0.75))-(df[i].quantile(0.25))
    ll=df[i].quantile(0.25)-(1.5*IQR)
    ul=df[i].quantile(0.75)+(1.5*IQR)
    df[i]=np.where(df[i]>ul,ul,np.where(df[i]<ll,ll,df[i]))
l=[]
for i in df.columns:
    l.append(df[i].var()==0)
df.drop(df.columns[[6,7]],axis=1,inplace=True)

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
plt.hist(df['Number of Referrals'])
plt.hist(df['Avg Monthly GB Download'])
plt.hist(df['Avg Monthly Long Distance Charges'])
plt.hist(df['Tenure in Months'])
plt.hist(df['Monthly Charge'])
plt.hist(df['Total Charges'])
plt.hist(df['Total Long Distance Charges'])
plt.hist(df['Total Revenue'])
#boxplot
plt.boxplot(df['Number of Referrals'])
plt.boxplot(df['Avg Monthly GB Download'])
plt.boxplot(df['Avg Monthly Long Distance Charges'])
plt.boxplot(df['Tenure in Months'])
plt.boxplot(df['Monthly Charge'])
plt.boxplot(df['Total Charges'])
plt.boxplot(df['Total Long Distance Charges'])
plt.boxplot(df['Total Revenue'])

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
dfnorm1=df_norm.iloc[:,[8,0,1,2,3,4,5,6,7]]
dfnorm1.head(10)
dfnorm1.iloc[:,1:4].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,4:7].groupby(dfnorm1.clust).mean()
dfnorm1.iloc[:,7:9].groupby(dfnorm1.clust).mean()
dfnorm1.to_csv("heirarchial3.csv", encoding = "utf-8")
'''cluster 3 has less revenue therefore high churn rate'''












