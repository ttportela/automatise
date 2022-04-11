# -*- coding: utf-8 -*-
'''
Automatize: Multi-Aspect Trajectory Data Mining Tool Library
The present application offers a tool, called AutoMATize, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATize integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
License GPL v.3 or superior

@author: Tarlis Portela
'''
import sys, os 
sys.path.insert(0, os.path.abspath(os.path.join('.')))
import glob2 as glob

# from automatize.main import importer
# importer(['S', 'mergeDatasets'], globals())
from automatize.run import moveResults

if len(sys.argv) < 1:
    print('Please run as:')
    print('\tMergeDatasets.py', 'PATH TO FOLDER')
    print('Example:')
    print('\tensemble-cls.py', '"./results/HiPerMovelets"')
    exit()

results_path = sys.argv[1]

dir_from = os.path.dirname(glob.glob(os.path.join(results_path, '**', 'train.csv'))[0])
# print(os.path.dirname(glob.glob(os.path.join(results_path, '**', 'train.csv'))[0]))

moveResults(dir_from, results_path)
