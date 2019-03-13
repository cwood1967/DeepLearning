import sys
import pandas as pd

def pickle_cluster(file, pk_file):
    df = pd.read_csv(file, usecols=["ClusterID", "File Name",
                                  "Index in File"],
                     low_memory=False)

    print(df.head())
    print(len(df))
    df.to_pickle(pk_file)

if __name__ == '__main__':
    cluster_file = sys.argv[1]
    pk_file = sys.argv[2]
    #cpath = '/Users/cjw/Projects/cjw/cyto/Clustering/Snail_ROP/UserDefined/'
    #pk_file = cpath + 'clusters.pkl'
    #cfile = cpath + 'ClusterIDs.csv'
    pickle_cluster(cluster_file, pk_file)
    print("Done")
