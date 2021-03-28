import os
import re
import numpy as np
import pandas as pd
import pickle
import json
import random
import threading
import getopt
import sys
import time
from scipy import sparse

#MAT A SPARSE
def unpack_A(dict_A,columns,Arows,typ = "Train"):
    """Converts the A dictionary to a numpy Matrix, and saves it as an npz type
    
    :param dict_A : dict
        Dictionary with the schema for building the A matrix
    :param columns : list
        list of unique methods across all smalli files
    :param Arows : list
        List of app names
    :param typ : str
        Either Train or Test, denoting the type of matrix being built
    """
    #MAT A SPARSE
    s = time.time()
    A = np.zeros([len(Arows), len(columns)], dtype = int)

    for i in dict_A.keys():
        t = Arows.index(i)
        chk = dict_A[i]
        for val in chk:
            if val in columns:
                A[t, columns.index(val)] = 1

    print(" Matrix A:  ", end ="")
    print(A.shape, end ="")
    print(" In " + str(int(time.time() - s)) + " Seconds")
    sparse.save_npz("src/matrices/"+typ+"_A.npz", sparse.csr_matrix(A), compressed=True)
    return True

#MAT B SPARSE
def unpack_B(dict_B,columns,typ = "Train"):
    """Converts the B dictionary to a numpy Matrix, and saves it as an npz type
    
    :param dict_B : dict
        Dictionary with the schema for building the B matrix
    :param columns : list
        list of unique methods across all smalli files
    :param typ : str
        Either Train or Test, denoting the type of matrix being built
    """
    s = time.time()
    B = np.zeros([len(columns), len(columns)], dtype = int)
    count = -1
    for i in dict_B.keys():
        #print("I",i)
        t = columns.index(i)
        chk = dict_B[i]
        for val in chk:
            if val in columns:
                B[t, columns.index(val)] = 1
                
    print(" Matrix B: ", end ="")
    print(B.shape, end ="")
    print(" In " + str(int(time.time() - s)) + " Seconds")
    sparse.save_npz("src/matrices/"+typ+"_B.npz", sparse.csr_matrix(B), compressed=True)
    return True

#MAT P SPARSE
def unpack_P(dict_P,columns,typ = "Train"):
    """Converts the P dictionary to a numpy Matrix, and saves it as an npz type
    
    :param dict_P : dict
        Dictionary with the schema for building the P matrix
    :param columns : list
        list of unique methods across all smalli files
    :param typ : str
        Either Train or Test, denoting the type of matrix being built
    """
    s = time.time()
    P = np.zeros([len(columns), len(columns)], dtype = int)

    for i in dict_P.keys():
        chk = dict_P[i]
        for val in chk:
            if val in columns:
                t = columns.index(val)
                for val2 in chk:
                    if val2 in columns:
                        P[t, columns.index(val2)] = 1
                
    print(" Matrix P: ", end ="")
    print(P.shape, end ="")
    print(" In " + str(int(time.time() - s)) + " Seconds")
    sparse.save_npz("src/matrices/"+typ+"_P.npz", sparse.csr_matrix(P), compressed=True)
    return True

#MAT I SPARSE
def unpack_I(dict_I,columns,typ = "Train"):
    """Converts the I dictionary to a numpy Matrix, and saves it as an npz type
    
    :param dict_I : dict
        Dictionary with the schema for building the I matrix
    :param columns : list
        list of unique methods across all smalli files
    :param typ : str
        Either Train or Test, denoting the type of matrix being built
    """
    s = time.time()
    I = np.zeros([len(columns), len(columns)], dtype = int)
    for i in dict_I.keys():
        chk = dict_I[i]
        for val in chk:
            if val in columns:
                t = columns.index(val)
                for val2 in chk:
                    if val2 in columns:
                        I[t, columns.index(val2)] = 1
        
    print(" Matrix I: ", end ="")
    print(I.shape, end ="")
    print(" In " + str(int(time.time() - s)) + " Seconds")
    sparse.save_npz("src/matrices/"+typ+"_I.npz", sparse.csr_matrix(I), compressed=True)
    return True