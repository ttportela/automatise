'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="automatise",
    version="0.1.b21",
    author="Tarlis Tortelli Portela",
    author_email="tarlis@tarlis.com.br",
    description="Automatise: Multiple Aspect Trajectory Data Mining Tool Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ttportela/automatise",
    packages=setuptools.find_packages(),
    scripts=[
        'scripts/MAT-Classifier-All.py',
        'scripts/MAT-Classifier-MLP.py',
        'scripts/MAT-Classifier-MLP_RF.py',
        'scripts/MAT-CheckRun.py',
        'scripts/MAT-ExportResults.py',
        'scripts/MAT-MergeDatasets.py',
        'scripts/MAT-PrintResults.py',
        'scripts/MAT-ResultsCheck.py',
        'scripts/MAT-ResultsTo.py',
        'marc/MARC.py',
        'pois/POIS.py',
        'pois/POIS-Classifier.py',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ),
    keywords='data mining, python, trajectory classification, trajectory analysis, movelets',
    license='GPL Version 3 or superior (see LICENSE file)',
)