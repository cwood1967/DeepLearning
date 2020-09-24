import numpy as np
cimport cython

def lookup(int index, str [:] imagelist):
    cdef str strindex
    cdef int i
    cdef int num
    strindex = "_" + str(index)+ "_"

    res = np.zeros((3,), dtype=np.int32)
    cdef int [::1] res_view = res

    for i in range(len(imagelist)):
        if strindex in imagelist[i]:
            res[num] = i
            num +=1
        if num >=3:
            return res
    return res