# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
# --------------------------------------------------------------------------------
# # ANALYSIS
from .main import importer #, display
importer(['S'], globals())

# --------------------------------------------------------------------------------
# from PACKAGE_NAME.Methods import Approach1, Approach2, ApproachRF, ApproachRFHP , ApproachMLP, ApproachDT, ApproachSVC
# --------------------------------------------------------------------------------
def ACC4All(res_path, prefix, save_results = True, modelfolder='model', classifiers=['MLP', 'RF', 'SVM'],
                   data_path=''):
#     import os
# #     import sys
#     import numpy as np
# #     import pandas as pd
#     import glob2 as glob
# #     from datetime import datetime
#     from ..main import importer
    importer(['S', 'glob'], globals())

    filelist = []
    filesList = []
    
    for files in glob.glob(os.path.join(res_path, prefix, "**", "*.txt")):
        fileName, fileExtension = os.path.splitext(files)
        method = os.path.basename(fileName)#[:-4]
        path = os.path.dirname(fileName)#[:-len(method)]
        todo = not os.path.exists( os.path.join(path, modelfolder) )
        empty = not os.path.exists( os.path.join(path, "train.csv") )
        if todo and not empty:
            ALL_Classifiers(path, '', '', save_results, modelfolder, classifiers, data_path)
        else:
            print(method + (" Done." if not empty else " Empty."))
# ----------------------------------------------------------------------------------
def ALL_Classifiers(res_path, prefix, dir_path, save_results = True, modelfolder='model', classifiers=['MLP', 'RF', 'SVM'],
                   data_path=''):
#     import os
# #     import sys
# #     import numpy as np
#     import pandas as pd
# #     import glob2 as glob
# #     from datetime import datetime
# #     def_random_seed(random_num, seed_num)
#     from ..main import importer
#     importer(['S'], locals())
    
    dir_path = os.path.join(res_path, prefix, dir_path)
    times = {'SVM': [0], 'RF': [0], 'MLP': [0], 'EC': [0]}
    
    times_file = os.path.join(dir_path, modelfolder, "classification_times.csv")
    if os.path.isfile(times_file):
        times = pd.read_csv(times_file)
    
    if 'RF' in classifiers:
        times['RF']  = [Classifier_RF(dir_path, save_results, modelfolder)]
    if 'MLP' in classifiers:
        times['MLP'] = [Classifier_MLP(dir_path, save_results, modelfolder)]
    if 'SVM' in classifiers:
        times['SVM'] = [Classifier_SVM(dir_path, save_results, modelfolder)]
    if 'EC' in classifiers:
        times['EC'] = [Classifier_EC(dir_path, data_path, save_results, modelfolder)]
#     t_svm = Classifier_SVM(dir_path, save_results, modelfolder)
#     t_rf  = Classifier_RF(dir_path, save_results, modelfolder)
#     t_mlp = Classifier_MLP(dir_path, save_results, modelfolder)
    
    # ------
    if (save_results) :
        if not os.path.exists(os.path.join(dir_path, modelfolder)):
            os.makedirs(os.path.join(dir_path, modelfolder))
        pd.DataFrame(times).to_csv(times_file)
            
# ----------------------------------------------------------------------------------
def MLP(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
#     import os
#     from ..main import importer
#     importer(['os'], locals())
#     def_random_seed(random_num, seed_num)
    
    dir_path = os.path.join(res_path, prefix, dir_path)
    t = Classifier_MLP(dir_path, save_results, modelfolder)
    return t

def RF(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
#     import os
#     from ..main import importer
#     importer(['os'], locals())
#     def_random_seed(random_num, seed_num)
    
    dir_path = os.path.join(res_path, prefix, dir_path)
    t = Classifier_RF(dir_path, save_results, modelfolder)
    return t

def SVM(res_path, prefix, dir_path, save_results = True, modelfolder='model'):
#     import os
#     from ..main import importer
#     importer(['os'], locals())
#     def_random_seed(random_num, seed_num)
    
    dir_path = os.path.join(res_path, prefix, dir_path)
    t = Classifier_SVM(dir_path, save_results, modelfolder)
    return t

# ----------------------------------------------------------------------------------
def Classifier_EC(dir_path, data_path, ensembles=None, save_results = True, modelfolder='model'):
#     import os
#     from PACKAGE_NAME.ensemble import ClassifierEnsemble
#     from ..main import importer
    importer(['os', 'TEC'], globals())
    # --------------------------------------------------------------------------------
    if ensembles == None:
        ensembles = {
            'marc':data_path,
            'movelets': dir_path,
        }
    # --------------------------------------------------------------------------------
    return ClassifierEnsemble(data_path, os.path.join(dir_path,'EC'), ensembles, 
                              save_results=save_results, modelfolder=modelfolder)

def Classifier_MLP(dir_path, save_results = True, modelfolder='model', X_train = None, y_train = None, X_test = None, y_test = None):
#     from datetime import datetime
#     # --------------------------------------------------------------------------------
#     from PACKAGE_NAME.Methods import Approach2
#     from ..main import importer
    importer(['datetime', 'A2'], globals())
    # --------------------------------------------------------------------------------
    if X_train is None:
        X_train, y_train, X_test, y_test = loadData(dir_path)
     
    # ---------------------------------------------------------------------------
    # Neural Network - Definitions:
    par_droupout = 0.5
    par_batch_size = 200
    par_epochs = 80
    par_lr = 0.00095
    
    # Building the neural network-
    print("Building neural network")
    lst_par_epochs = [80,50,50,30,20]
    lst_par_lr = [0.00095,0.00075,0.00055,0.00025,0.00015]
    
    time = datetime.now()
    Approach2(X_train, y_train, X_test, y_test, par_batch_size, lst_par_epochs, lst_par_lr, par_droupout, save_results, dir_path, modelfolder)
    time = (datetime.now()-time).total_seconds() * 1000
    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time

# ----------------------------------------------------------------------------------
def Classifier_RF(dir_path, save_results = True, modelfolder='model', X_train = None, y_train = None, X_test = None, y_test = None):
#     from datetime import datetime
#     # --------------------------------------------------------------------------------
#     from PACKAGE_NAME.Methods import ApproachRF
#     from ..main import importer
    importer(['datetime', 'ARF'], globals())
    # --------------------------------------------------------------------------------
    if X_train is None:
        X_train, y_train, X_test, y_test = loadData(dir_path)
    
    # ---------------------------------------------------------------------------
    # Random Forest - Definitions:
    # Este experimento eh para fazer uma varredura de arvores em random forestx
    #n_estimators = np.arange(10, 751, 10)
    #n_estimators = np.append([1], n_estimators)
    n_estimators = [300]
    print(n_estimators)
    
    print("Building random forest models")
    time = datetime.now()
    ApproachRF(X_train, y_train, X_test, y_test, n_estimators, save_results, dir_path, modelfolder)
    time = (datetime.now()-time).total_seconds() * 1000
    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")
    return time
    
# ----------------------------------------------------------------------------------
def Classifier_SVM(dir_path, save_results = True, modelfolder='model', X_train = None, y_train = None, X_test = None, y_test = None):
#     from datetime import datetime
#     # --------------------------------------------------------------------------------
#     from PACKAGE_NAME.Methods import ApproachSVC
#     from ..main import importer
    importer(['datetime', 'ASVC'], globals())
    # --------------------------------------------------------------------------------
    if X_train is None:
        X_train, y_train, X_test, y_test = loadData(dir_path)
    
    print("Building SVM models")
    time = datetime.now()
    ApproachSVC(X_train, y_train, X_test, y_test, save_results, dir_path, modelfolder)
    time = (datetime.now()-time).total_seconds() * 1000
    # ---------------------------------------------------------------------------------
    print("Done. " + str(time) + " milliseconds")
    print("---------------------------------------------------------------------------------")

# --------------------------------------------------------------------------------->  
# Importing the dataset
def loadData(dir_path):
#     import os
# #     import sys
# #     import numpy as np
#     import pandas as pd
# #     import glob2 as glob
# #     from datetime import datetime

#     from sklearn import preprocessing
#     from ..main import importer
    importer(['S', 'MinMaxScaler'], globals())
    
    print("Loading train and test data from... " + dir_path)
    dataset_train = pd.read_csv(os.path.join(dir_path, "train.csv"))
    dataset_test  = pd.read_csv(os.path.join(dir_path, "test.csv"))
#     n_jobs = N_THREADS
    print("Done.")

    nattr = len(dataset_train.iloc[1,:])
    print("Number of attributes: " + str(nattr))
    
    if (len( set(dataset_test.columns).symmetric_difference(set(dataset_train.columns)) ) > 0):
        print('*ERROR* Divergence in train and test columns:', 
              len(dataset_train.columns), 'train and', len(dataset_test.columns), 'test')

    # Separating attribute data (X) than class attribute (y)
    X_train = dataset_train.iloc[:, 0:(nattr-1)].values
    y_train = dataset_train.iloc[:, (nattr-1)].values

    X_test = dataset_test.iloc[:, 0:(nattr-1)].values
    y_test = dataset_test.iloc[:, (nattr-1)].values

    # Replace distance 0 for presence 1
    # and distance 2 to non presence 0
    X_train[X_train == 0] = 1
    X_train[X_train == 2] = 0
    X_test[X_test == 0] = 1
    X_test[X_test == 2] = 0
    
    # Scaling data
#     min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test