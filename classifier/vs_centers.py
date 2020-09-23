
#%%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans


#%%
#### this is to get the features used for clustering
df = pd.read_csv('/Users/cjw/Data/cyto/Classifier/ClusterIDs.csv')
df.head()


#%%
data = df.iloc[:,6:].values


#%%
ag = AgglomerativeClustering(n_clusters=7)
cag = ag.fit_predict(data)
km = KMeans(n_clusters=7)
ckm = km.fit_predict(data)


#%%
df['ag'] = cag
df['km'] = ckm


#%%
sns.countplot(x='ag', data=df)


#%%
cemb = umap.UMAP(n_neighbors=10, n_components=7, min_dist=.05).fit_transform( data)

#%%
ag2 = AgglomerativeClustering(n_clusters=6)
cc = ag2.fit_predict(cemb)
df['cc'] = cc
cemb.shape


#%%
emb = umap.UMAP(n_neighbors=10, min_dist=.05).fit_transform( data)
emb.shape


#%%
df['ux'] = emb[:,0]
df['uy'] = emb[:,1]
df['size'] = 1


#%%
plt.figure(figsize=(8,8))
cmap = sns.cubehelix_palette(dark=.01, light=.9, as_cmap=True)
sns.scatterplot(x='ux', y='uy',hue='cc', palette='Set1', edgecolor='none', s=10, marker='o', legend="full", data=df)

#%% [markdown]
# Just use the column 'cc' to create the labels like i did before

#%%
from sklearn.preprocessing import LabelBinarizer

labels = df['cc']
nn = list(np.arange(labels.min(), labels.max() + 1))
print(nn)
b = LabelBinarizer()
b.fit(nn)

p = b.transform(labels)
p.shape


#%%
np.savetxt('cc_6.csv', cc)


#%%
mm = np.memmap('.' + 'cc_6_labels.mm', dtype=np.float32, mode='w+', shape=p.shape)
mm[:] = pz
mm.flush()
del mm


#%%
from pygraphml import GraphMLParser
from pygraphml import Graph


#%%



#%%
fdl = pd.read_csv('FDL_coords.csv', delimiter=';')
fdl['Y'] = -fdl['Y']
fdl.head()


#%%
res = fdl.set_index('EventID').join(df.set_index('EventID'))
print(len(res))
res.head()


#%%
plt.figure(figsize=(8,8))
sns.scatterplot(x='X', y='Y', hue='cc', palette='Set1', edgecolor='none',
                s=12, marker='o', legend="full", data=res)


#%%
plt.figure(figsize=(8,8))
sns.scatterplot(x='X', y='Y', hue='ClusterID', palette='Set1', edgecolor='none', s=12, marker='o', legend="full", data=res)


#%%
all_sm = np.load('Data/all_pickle.pkl.npy')


#%%
predictions = np.argmax(all_sm, axis=1)
predictions.shape


#%%
sns.distplot(predictions)


#%%
df['predictions'] = predictions


#%%
df.head()


#%%



#%%
resp = fdl.set_index('EventID').join(df.set_index('EventID'))
print(len(resp))
resp.head()


#%%
plt.figure(figsize=(12,12))
sns.scatterplot(x='X', y='Y', hue='predictions', palette='Set1', edgecolor='none', s=12, marker='o', legend="full", data=resp)

#%%
plt.figure(figsize=(12,12))
sns.scatterplot(x='ux', y='uy',
                hue='predictions', palette='Set1', edgecolor='none',
                s=12, marker='o', legend="full", data=df)

plt.figure(figsize=(12,12))
sns.scatterplot(x='ux', y='uy',
                hue='ClusterID', palette='Set1', edgecolor='none',
                s=12, marker='o', legend="full", data=df)
#%%
resp['prediction'] = predictions[resp.index]
plt.figure(figsize=(8,8))
sns.scatterplot(x='ux', y='uy', hue='predictions', palette='Set1',
                edgecolor='none', s=4, marker='o', legend="full", data=df)
df.head()

#%%
predictions.shape
#%%
train_ac_df = pd.read_csv('Training_csv/run-train-tag-accuracy_1.csv')
test_ac_df = pd.read_csv('Training_csv/run-test-tag-accuracy_1.csv')
train_ac_df['run'] = len(train_ac_df)*['train']
test_ac_df['run'] = len(test_ac_df)*['test']
ac_df = pd.concat([train_ac_df, test_ac_df], axis=0)

train_loss_df = pd.read_csv('Training_csv/run-train-tag-loss.csv')
test_loss_df = pd.read_csv('Training_csv/run-test-tag-loss.csv')
train_loss_df['run'] = len(train_loss_df)*['train']
test_loss_df['run'] = len(test_loss_df)*['test']
loss_df = pd.concat([train_loss_df, test_loss_df], axis=0)
loss_df.head(), ac_df.tail()


#%%
sns.relplot(x='Step', y='Value', hue='run', style='run', data=ac_df, kind='line')
plt.savefig('Training_csv/acc.pdf')


#%%
sns.relplot(x='Step', y='Value', hue='run', style='run', data=loss_df, kind='line')
plt.savefig('Training_csv/loss.pdf')


#%%
from sklearn.manifold import TSNE

tsne = TSNE().fit_transform(data)



#%%
df['tx'] = tsne[:,0]
df['ty'] = tsne[:,1]
with plt.style.context("fast", True):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='tx', y='ty', hue='predictions', palette='Set1',
                    edgecolor='none', s=2, marker='o', legend="full", data=df)

plt.legend(loc='upper left', prop={'size':6}, bbox_to_anchor=(1,1))

#%%

cdmap = {156:0, 157:1, 158:2, 159:3,
         160:4, 161:5, 162:6, 163:4, 164:0}

df['ClusterIDMap'] = df['ClusterID'].map(cdmap)

#%%
with plt.style.context("fast", True):
    plt.figure(figsize=(6,6))
    sns.scatterplot(x='tx', y='ty', hue='ClusterIDMap', palette='Set1',
                    edgecolor='none', s=2, marker='o', legend="full", data=df)
    plt.legend(loc='upper left', prop={'size':10}, bbox_to_anchor=(1,1))

#%% 
print(plt.style.available)

#%%
