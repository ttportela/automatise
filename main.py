# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
'''
# def importer(key=['S'], this=None):
#     import sys 
#     if sys.version_info >= (3,9):
#         importer39(key, this)
#     else:
#         importer(key, this)

def importer(key=['S'], this=None):
    import importlib
    automatize_name = 'automatize'
    
    mdic = {}
    # Progress Bar:
    if set(key) & set(['*', 'tqdm']):
        module = importlib.import_module('tqdm.auto')
        mdic.update( {'tqdm': getattr(module, 'tqdm')} )
            
    # Standard:
    if set(key) & set(['*', 'S', 's', 'pd', 'os', 'np']):
        if set(key) & set(['*', 'S', 's', 'pd']):
            module = importlib.import_module('pandas')
            mdic.update( {'pd': module} )
        if set(key) & set(['*', 'S',  's', 'os']):
            module = importlib.import_module('os')
            mdic.update( {'os': module} )
        if set(key) & set(['*', 'S', 'np']):
            module = importlib.import_module('numpy')
            mdic.update( {'np': module} )
       
    if set(key) & set(['*', 'general', 'NN', 'MLP', 'sys', 'argmax', 'zip', 'random', 'shutil']):
        if set(key) & set(['*', 'sys']):
            module = importlib.import_module('sys')
            mdic.update( {'sys': module} )
        if set(key) & set(['*', 'argmax', 'NN', 'MLP']):
            module = importlib.import_module('numpy')
            mdic.update( {'argmax': getattr(module, 'argmax')} )
        if set(key) & set(['*', 'zip']):
            module = importlib.import_module('zipfile')
            mdic.update( {'ZipFile': getattr(module, 'ZipFile')} )
        if set(key) & set(['*', 'random']):
            module = importlib.import_module('random')
            mdic.update( {'random': module} )
        if set(key) & set(['*', 'shutil']):
            module = importlib.import_module('shutil')
            mdic.update( {'shutil': module} )

    if set(key) & set(['*', 'datetime', 'dateparser']):
        if set(key) & set(['*', 'datetime']):
            module = importlib.import_module('datetime')
            mdic.update( {'datetime': getattr(module, 'datetime')} )
        if set(key) & set(['*', 'dateparser']):
            module = importlib.import_module('dateutil.parser')
            mdic.update( {'dateparser': module} )
            
    if set(key) & set(['*', 'mat']):
        module = importlib.import_module('matplotlib')
        mdic.update( {'mat': module} )
    if set(key) & set(['*', 'G', 'plt']):
        module = importlib.import_module('matplotlib.pyplot')
        mdic.update( {'plt': module} )
        
    if set(key) & set(['*', 'G', 'plotly']):
        module = importlib.import_module('plotly')
        mdic.update( {'plotly': module} )
        mdic.update( {'go': getattr(module, 'graph_objects')} )
        
    if set(key) & set(['*', 'nx']):
        module = importlib.import_module('networkx')
        mdic.update( {'nx': module} )
        
    if set(key) & set(['*', 'json']):
        module = importlib.import_module('json')
        mdic.update( {'json': module} )
        
    # 3rd Party:
    if set(key) & set(['*', 'others', 'glob', 'glob2', 'tf', 'tensorflow', 'gh']):
        if set(key) & set(['*', 'others', 'glob', 'glob2']):
            module = importlib.import_module('glob2')
            mdic.update( {'glob': module} )

        if set(key) & set(['*', 'others', 'tf', 'tensorflow']):
            module = importlib.import_module('tensorflow')
            mdic.update( {'tf': module} )   

        if set(key) & set(['*', 'others', 'gh']):
            module = importlib.import_module('geohash2')
            mdic.update( {'gh': module} )   
            
        if set(key) & set(['*', 'others', 're']):
            module = importlib.import_module('re')
            mdic.update( {'re': module} )  
    
    # FOR neural networks:
    if set(key) & set(['*', 'pynn', 'models', 'layers', 'initializers', 'regularizers', 'callbacks', 'optimizers', 
                       'NN', 'MLP', 'MARC', 'POIS', 'Sequential', 'Model', 'Dense', 'Dropout', 'LSTM', 'GRU', 'Input', 
                       'Add', 'Average', 'Concatenate', 'Embedding', 'Activation', 
                       'he_uniform', 'l1', 'l2', 'EarlyStopping', 'History', 'Adam', 'to_categorical']):
        
        if set(key) & set(['*', 'models', 'NN', 'MLP', 'MARC', 'POIS', 'Sequential', 'Model']):
            module = importlib.import_module('tensorflow.keras.models')
            if set(key) & set(['*', 'models', 'NN', 'MLP', 'POIS', 'Sequential']):
                mdic.update( {'Sequential': getattr(module, 'Sequential')} )
            if set(key) & set(['*', 'models', 'MARC', 'Model']):
                mdic.update( {'Model': getattr(module, 'Model')} )

        if set(key) & set(['*', 'layers', 'NN', 'MLP', 'MARC', 'POIS', 
                           'LSTM', 'Dense', 'Dropout', 'GRU', 'Input', 'Activation',
                           'Add', 'Average', 'Concatenate', 'Embedding']):
            module = importlib.import_module('tensorflow.keras.layers')
            if set(key) & set(['*', 'layers', 'NN', 'MLP', 'MARC', 'POIS', 'Dense']):
                mdic.update( {'Dense': getattr(module, 'Dense')} )
            if set(key) & set(['*', 'layers', 'NN', 'MLP', 'MARC', 'POIS', 'Dropout']):
                mdic.update( {'Dropout': getattr(module, 'Dropout')} )

            if set(key) & set(['*', 'layers', 'MARC', 'LSTM']):
                mdic.update( {'LSTM': getattr(module, 'LSTM')} )
            if set(key) & set(['*', 'layers', 'MARC', 'GRU']):
                mdic.update( {'GRU': getattr(module, 'GRU')} )
            if set(key) & set(['*', 'layers', 'MARC', 'Input']):
                mdic.update( {'Input': getattr(module, 'Input')} )
            if set(key) & set(['*', 'layers', 'MARC', 'Add']):
                mdic.update( {'Add': getattr(module, 'Add')} )
            if set(key) & set(['*', 'layers', 'MARC', 'Average']):
                mdic.update( {'Average': getattr(module, 'Average')} )
            if set(key) & set(['*', 'layers', 'MARC', 'Concatenate']):
                mdic.update( {'Concatenate': getattr(module, 'Concatenate')} )
            if set(key) & set(['*', 'layers', 'MARC', 'Embedding']):
                mdic.update( {'Embedding': getattr(module, 'Embedding')} )

            if set(key) & set(['*', 'layers', 'Activation']):
                mdic.update( {'Activation': getattr(module, 'Activation')} )

        if set(key) & set(['*', 'initializers', 'MARC', 'he_uniform']):
#             module = importlib.import_module('keras.initializers')
            module = importlib.import_module('tensorflow.keras.initializers')
            mdic.update( {'he_uniform': getattr(module, 'he_uniform')} )

        if set(key) & set(['*', 'regularizers', 'NN', 'MLP', 'MARC', 'POIS', 'l1', 'l2']):
#             module = importlib.import_module('keras.regularizers')
            module = importlib.import_module('tensorflow.keras.regularizers')
            if set(key) & set(['*', 'regularizers', 'MARC', 'l1']):
                mdic.update( {'l1': getattr(module, 'l1')} )
            if set(key) & set(['*', 'regularizers', 'POIS', 'l2']):
                mdic.update( {'l2': getattr(module, 'l2')} )

            if set(key) & set(['*', 'regularizers', 'NN', 'MLP']):
#                 module = importlib.import_module('tensorflow.keras')
#                 mdic.update( {'regularizers': getattr(module, 'regularizers')} )
#                 module = importlib.import_module('tensorflow.keras.regularizers')
                mdic.update( {'regularizers': module} )

        if set(key) & set(['*', 'callbacks', 'MARC', 'POIS', 'EarlyStopping', 'History']):
#             module = importlib.import_module('keras.callbacks')
            module = importlib.import_module('tensorflow.keras.callbacks')
            if set(key) & set(['*', 'callbacks', 'MARC', 'POIS', 'EarlyStopping']):
                mdic.update( {'EarlyStopping': getattr(module, 'EarlyStopping')} )
            if set(key) & set(['*', 'callbacks', 'POIS', 'History']):
                mdic.update( {'History': getattr(module, 'History')} )

        if set(key) & set(['*', 'optimizers', 'NN', 'MLP', 'MARC', 'POIS', 'Adam']):
            module = importlib.import_module('tensorflow.keras.optimizers')
            mdic.update( {'Adam': getattr(module, 'Adam')} )

        if set(key) & set(['*', 'NN', 'MLP', 'to_categorical']):
            module = importlib.import_module('tensorflow.keras.utils')
            mdic.update( {'to_categorical': getattr(module, 'to_categorical')} )
        
    # FOR sklearn:
    if set(key) & set(['*', 'classifiers', 'RF', 'RFHP', 'DT', 'SVC', 'KerasClassifier']):
        if set(key) & set(['*', 'classifiers', 'RF', 'RFHP']):
            module = importlib.import_module('sklearn.ensemble')
            if set(key) & set(['*', 'classifiers', 'RF', 'RandomForestClassifier']):
                mdic.update( {'RandomForestClassifier': getattr(module, 'RandomForestClassifier')} )

        if set(key) & set(['*', 'classifiers', 'RFHP']):
            module = importlib.import_module('sklearn.model_selection')
            mdic.update( {'RandomizedSearchCV': getattr(module, 'RandomizedSearchCV')} )

        if set(key) & set(['*', 'classifiers', 'DT']):
            module = importlib.import_module('sklearn.tree')
            mdic.update( {'DecisionTreeClassifier': getattr(module, 'DecisionTreeClassifier')} )

        if set(key) & set(['*', 'classifiers', 'SVC']):
            module = importlib.import_module('sklearn.svm')
            mdic.update( {'SVC': getattr(module, 'SVC')} )

        if set(key) & set(['*', 'classifiers', 'KerasClassifier']):
#             module = importlib.import_module('keras.wrappers.scikit_learn')
            module = importlib.import_module('tensorflow.keras.wrappers.scikit_learn')
            mdic.update( {'KerasClassifier': getattr(module, 'KerasClassifier')} )
        
#     if set(key) & set(['*', 'metrics']):
#         module = importlib.import_module('tensorflow.keras.metrics')
#         mdic.update( {'metrics': module} )
    if set(key) & set(['*', 'metrics', 'precision_score', 'recall_score', 'f1_score', 'accuracy_score']):
        module = importlib.import_module('sklearn.metrics')
        if set(key) & set(['*', 'metrics', 'precision_score']):
            mdic.update( {'precision_score': getattr(module, 'precision_score')} )
        if set(key) & set(['*', 'metrics', 'recall_score']):
            mdic.update( {'recall_score': getattr(module, 'recall_score')} )
        if set(key) & set(['*', 'metrics', 'f1_score']):
            mdic.update( {'f1_score': getattr(module, 'f1_score')} )
        if set(key) & set(['*', 'metrics', 'accuracy_score']):
            mdic.update( {'accuracy_score': getattr(module, 'accuracy_score')} )
        
    if set(key) & set(['*', 'report', 'NN', 'MLP', 'RF', 'RFHP', 'DT', 'SVC']):
        module = importlib.import_module('sklearn.metrics')
        mdic.update( {'classification_report': getattr(module, 'classification_report')} )      
      
    if set(key) & set(['*', 'preprocessing', 'encoding', 'PP', 'NN', 'MLP', 'MARC', 'POIS', 'MinMaxScaler', 
                       'KFold', 'split', 'train_test_split', 'pad_sequences', 'LabelEncoder', 'OneHotEncoder', 
                       'bin_geohash', 'geohash']):
        if set(key) & set(['*', 'preprocessing', 'PP', 'NN', 'MLP', 'POIS']):
            module = importlib.import_module('sklearn.preprocessing')
            mdic.update( {'preprocessing': module} )
#             module = importlib.import_module('sklearn')
#             mdic.update( {'preprocessing': getattr(module, 'preprocessing')} )
        if set(key) & set(['*', 'preprocessing', 'MinMaxScaler']):
            module = importlib.import_module('sklearn.preprocessing')
            mdic.update( {'MinMaxScaler': getattr(module, 'MinMaxScaler')} )

        if set(key) & set(['*', 'preprocessing', 'KFold', 'split', 'train_test_split']):
            module = importlib.import_module('sklearn.model_selection')
            if set(key) & set(['*', 'preprocessing', 'KFold']):
                mdic.update( {'KFold': getattr(module, 'KFold')} )
            if set(key) & set(['*', 'preprocessing', 'split', 'train_test_split']):
                mdic.update( {'train_test_split': getattr(module, 'train_test_split')} )

        if set(key) & set(['*', 'preprocessing', 'encoding', 'MARC', 'pad_sequences']):
#             module = importlib.import_module('keras.preprocessing.sequence')
            module = importlib.import_module('tensorflow.keras.preprocessing.sequence')
            mdic.update( {'pad_sequences': getattr(module, 'pad_sequences')} )

        if set(key) & set(['*', 'preprocessing', 'encoding', 'LabelEncoder', 'OneHotEncoder']):
            module = importlib.import_module('sklearn.preprocessing')
            mdic.update( {'LabelEncoder': getattr(module, 'LabelEncoder'), 'OneHotEncoder': getattr(module, 'OneHotEncoder')} )

        if set(key) & set(['*', 'preprocessing', 'encoding', 'bin_geohash', 'geohash']):
            module = importlib.import_module(automatize_name+'.marc.core.utils')
            if set(key) & set(['*', 'preprocessing', 'encoding', 'bin_geohash']):
                mdic.update( {'bin_geohash': getattr(module, 'bin_geohash')} )
            if set(key) & set(['*', 'preprocessing', 'geohash']):
                mdic.update( {'geohash': getattr(module, 'geohash')} )
        
    if set(key) & set(['*', 'K']):
        module = importlib.import_module('tensorflow.keras.backend')
        mdic.update( {'K': module} )
                
    if set(key) & set(['*', 'classifiers', 'A1', 'Approach1', 'A2', 'Approach2', 'ARF', 'ApproachRF', 'ARFHP', 'ApproachRFHP', 
                       'ASVC', 'ApproachSVC', 'ADT', 'ApproachDT', 'AMLP', 'ApproachMLP']):
        module = importlib.import_module(automatize_name+'.classifiers')
        if set(key) & set(['*', 'classifiers', 'A1', 'Approach1']):
            mdic.update( {'Approach1': getattr(module, 'Approach1')} )
        if set(key) & set(['*', 'classifiers', 'A2', 'Approach2']):
            mdic.update( {'Approach2': getattr(module, 'Approach2')} )
        if set(key) & set(['*', 'classifiers', 'ARF', 'ApproachRF']):
            mdic.update( {'ApproachRF': getattr(module, 'ApproachRF')} )
        if set(key) & set(['*', 'classifiers', 'ARFHP', 'ApproachRFHP']):
            mdic.update( {'ApproachRFHP': getattr(module, 'ApproachRFHP')} )
        if set(key) & set(['*', 'classifiers', 'ASVC', 'ApproachSVC']):
            mdic.update( {'ApproachSVC': getattr(module, 'ApproachSVC')} )
        if set(key) & set(['*', 'classifiers', 'ADT', 'ApproachDT']):
            mdic.update( {'ApproachDT': getattr(module, 'ApproachDT')} )
        if set(key) & set(['*', 'classifiers', 'AMLP', 'ApproachMLP']):
            mdic.update( {'ApproachMLP': getattr(module, 'ApproachMLP')} )  
            
    if set(key) & set(['*', 'report', 'f1', 'classification_report_csv', 'calculateAccTop5']): #, 'classification_report'
        module = importlib.import_module(automatize_name+'.classifiers')
#         if set(key) & set(['*', 'report', 'classification_report']):
#             mdic.update( {'classification_report': getattr(module, 'classification_report')} )
        if set(key) & set(['*', 'report', 'classification_report_csv']):
            mdic.update( {'classification_report_csv': getattr(module, 'classification_report_csv')} )
        if set(key) & set(['*', 'report', 'calculateAccTop5']):
            mdic.update( {'calculateAccTop5': getattr(module, 'calculateAccTop5')} )
        if set(key) & set(['*', 'report', 'f1']):
            mdic.update( {'f1': getattr(module, 'f1')} )
            
            
    if set(key) & set(['*', 'results', 'STATS', 'printLatex', 'results2df', 'check_run'
                       'containErrors', 'containWarnings', 'containTimeout', 'get_stats',
                       'format_stats', 'format_hour']):
        module = importlib.import_module(automatize_name+'.results')
        if set(key) & set(['*', 'results', 'STATS']):
            mdic.update( {'STATS': getattr(module, 'STATS')} )
        if set(key) & set(['*', 'results', 'printLatex']):
            mdic.update( {'printLatex': getattr(module, 'printLatex')} )
        if set(key) & set(['*', 'results', 'results2df']):
            mdic.update( {'results2df': getattr(module, 'results2df')} )
        if set(key) & set(['*', 'results', 'check_run', 'containErrors', 'containWarnings', 'containTimeout']):
            mdic.update( {'containErrors': getattr(module, 'containErrors'), 
                          'containWarnings': getattr(module, 'containWarnings'), 
                          'containTimeout': getattr(module, 'containTimeout')} )
        if set(key) & set(['*', 'results', 'format_stats']):
            mdic.update( {'format_stats': getattr(module, 'format_stats')} )
        if set(key) & set(['*', 'format_hour']): # except
            mdic.update( {'format_hour': getattr(module, 'format_hour')} )
        if set(key) & set(['*', 'results', 'get_stats']):
            mdic.update( {'get_stats': getattr(module, 'get_stats')} )
            
            
    if set(key) & set(['*', 'analysis', 'loadData']):
        module = importlib.import_module(automatize_name+'.analysis')
        if set(key) & set(['*', 'analysis', 'loadData']):
            mdic.update( {'loadData': getattr(module, 'loadData')} )
            
            
    if set(key) & set(['*', 'run', 'methods', 'mergeDatasets']):
        module = importlib.import_module(automatize_name+'.run')
        if set(key) & set(['*', 'run', 'mergeDatasets']):
            mdic.update( {'mergeDatasets': getattr(module, 'mergeDatasets')} )
        if set(key) & set(['*', 'run', 'methods']): #MARC, POIFREQ
            mdic.update( {'Movelets': getattr(module, 'Movelets')} )
            mdic.update( {'MARC': getattr(module, 'MARC')} )
            mdic.update( {'POIFREQ': getattr(module, 'POIFREQ')} )
            
    if set(key) & set(['*', 'movelets', 'markov', 'anytree', 'Digraph']):
        if set(key) & set(['*', 'movelets', 'markov', 'Digraph']):
            module = importlib.import_module('graphviz')
            mdic.update( {'Digraph': getattr(module, 'Digraph')} ) 
            
        if set(key) & set(['*', 'movelets', 'anytree']):
            module = importlib.import_module('anytree')
            mdic.update( {'Node': getattr(module, 'Node')} )
            mdic.update( {'RenderTree': getattr(module, 'RenderTree')} )
            mdic.update( {'DotExporter': getattr(importlib.import_module('anytree.exporter'), 'DotExporter')} )
            
        if set(key) & set(['*', 'movelets', 'markov']):
            module = importlib.import_module('graphviz')
            mdic.update( {'Digraph': getattr(module, 'Digraph')} )
    
    if set(key) & set(['*', 'io', 'ts_io', 'load_from_tsfile_to_dataframe', 'readDataset']):
        if set(key) & set(['*', 'io', 'readDataset']):
            module = importlib.import_module(automatize_name+'.preprocessing')
#             if set(key) & set(['*', 'io', 'readDataset']):
            mdic.update( {'readDataset': getattr(module, 'readDataset')} )
    
        if set(key) & set(['*', 'ts_io', 'load_from_tsfile_to_dataframe']):
            module = importlib.import_module(automatize_name+'.io.ts_io')
#             if set(key) & set(['*', 'ts_io', 'load_from_tsfile_to_dataframe']):
            mdic.update( {'load_from_tsfile_to_dataframe': getattr(module, 'load_from_tsfile_to_dataframe')} )
        
    if this is not None:
        this.update(mdic)
#         unpack(this, mdic)
        
    return mdic
    
# def unpack(this, modules):
#     this.update(modules)
    
# ------------------------------------------------------
def def_random_seed_compat(random_num=1, seed_num=1):
    # Para garantir reprodutibilidade
    from numpy.random import seed
    import tensorflow # import set_random_seed
    seed(seed_num)
    tensorflow.compat.v1.set_random_seed(random_num)
    
def def_random_seed(random_num=1, seed_num=1):
    # Para garantir reprodutibilidade
    from numpy.random import seed
    import tensorflow # import set_random_seed
    seed(seed_num)
#     tensorflow.random.set_seed(random_num)
    tensorflow.set_random_seed(random_num)
    
def display(df):
    import pandas as pd
#     importer(['pd'], locals())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        if isConsole():
            print(df)
        else:
            from IPython.display import display as dy
            dy(df)

def isConsole():
    try:
        __IPYTHON__
        _in_ipython_session = False
    except NameError:
        _in_ipython_session = True
    return _in_ipython_session