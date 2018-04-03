import os
import glob

import numpy as np
import pandas as pd


def read_data_file(datafile):
    names = ['id', 'fid', 'file', 'mmfile', 'plate', 'row', 'column',
             'field', 'yc', 'xc']

    df = pd.read_csv(datafile, sep=",", names=names)
    mmfile = df['mmfile']
    maxrow = df['row'].max()
    well = maxrow*(df['column'] - 1) + df['row'] - 1
    df['well'] = well
    mmfile = mmfile.str.replace("/Users/cjw/", '')
    df['mmfile'] = mmfile
    return df

def clean_mmfilename(df):
    mmcol = df['mmfile']

def getXY(mmdict, df, rowid, size):
    rowd = df[df['id'] == rowid]
    row = rowd.iloc[0]
    fid = int(row['fid'])
    xc = row['xc']
    yc = row['yc']
    well = row['well']
    mfile = mmdict[row['mmfile'].strip()]
    x = int(xc) - size//2
    y = int(yc) - size//2

    shape = mfile.shape
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > (shape[2] - size):
        x = shape[2] - size
    if y > (shape[1] - size):
        y = shape[1] - size

    return mfile, fid, x, y, well


def getbatch(mmdict, df, start, batchsize, size, nchannels, channels=None):
    if channels is None:
        channels = np.arange(nchannels)
    dfsize = len(df)
    dx = dfsize//batchsize
    rownums = np.linspace(start, start + dx*(batchsize - 1), batchsize,
                          dtype=np.int32)
    #print(len(rownums))
    #print(rownums)
    batch = np.zeros((batchsize, size, size, nchannels))
    wells = []
    for i, v in enumerate(rownums):
        mfile, fid, x, y, well = getXY(mmdict, df, v,  size)
        z = mfile[fid][y:y + size, x:x + size, channels]
        wells.append(well)
        batch[i] = z
    wells = np.asarray(wells)
    return batch, wells, rownums


def getWell(mmdict, df, size, row, column, nchannels, channels=None):
    if channels is None:
        channels = np.arange(nchannels)

    welldf = df[df['row'] == row]
    welldf = welldf[welldf['column'] == column]
    dfsize = len(welldf)

    images = np.zeros((dfsize, size, size, nchannels))
    for i, (index, irow) in enumerate(welldf.iterrows()):
        rid = irow['id']
        fid = irow['fid']
        xc = irow['xc']
        yc = irow['yc']
        x = int(xc) - size // 2
        y = int(yc) - size // 2
        mfile = mmdict[irow['mmfile'].strip()]

        shape = mfile.shape
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > (shape[2] - size):
            x = shape[2] - size
        if y > (shape[1] - size):
            y = shape[1] - size

        z = mfile[fid][y:y + size, x:x + size, channels]
        images[i] = z
    return images

def create_mm_dataframe(mmfiles_dict):
    pass

def list_mmfiles(dir):
    if not dir.endswith("/"):
        dir += "/"
        
    mmfiles = glob.glob(dir + "*" + ".mm")
    return mmfiles