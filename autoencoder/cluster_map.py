import pandas as pd

def pickle_cluster(file, pk_file):
    df = pd.read_csv(file, usecols=["ClusterID", "File Name",
                                  "Index in File"],
                     low_memory=False)

    print(df.head())
    print(len(df))
    df.to_pickle(pk_file)


cpath = '/Users/cjw/Projects/cjw/cyto/Clustering/DeepLearning/ClusteringResults/'
pk_file = cpath + 'clusters.pkl'
cfile = cpath + 'ClusterIDs.csv'
pickle_cluster(cfile, pk_file)
