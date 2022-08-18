# Automatise: Multiple Aspect Trajectory Data Mining Tool Library
---

\[[Publication](#)\] \[[citation.bib](citation.bib)\] \[[GitHub](https://github.com/ttportela/automatise)\] \[[PyPi](https://pypi.org/project/automatise/)\]


Welcome to Automatise Framework for Multiple Aspect Trajectory Analysis. You can use it as a web-platform or a Python library.

The present application offers a tool, called AutoMATise, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. The AutoMATise integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

### Main Modules

- [Datasets](/datasets): Datasets descriptions, statistics and files to download;
- [Methods](/methods): Methods for trajectory classification and movelet extraction;
- [Scripting](/experiments): Script generator for experimental evaluation on available methods (Linux shell);
- [Results](/results): Experiments on trajectory datasets and method rankings;
- [Analysis](/analysis): Multiple Aspect Trajectory Analysis Tool (trajectory and movelet visualization);
- [Publications](/publications): Multiple Aspect Trajectory Analysis related publications;
- [Tutorial](/tutorial): Tutorial on how to use Automatise as a Python library.


### Installation

Install directly from PyPi repository, or, download from github. Intalling with pip will also provide command line scripts (available in folder `automatise/scripts`). (python >= 3.5 required)

```bash
    pip install automatise
```

To use Automatise as a python library, find examples in this sample Jupyter Notebbok: [Automatise_Sample_Code.ipynb](./assets/examples/Automatise_Sample_Code.ipynb)

### Web Application

Run as wep application by running the script:

```bash
    python3 automatise/app.py
```

### Citing

If you use `automatise` please cite the following paper:

    Portela, Tarlis Tortelli; Bogorny, Vania; Bernasconi, Anna; Renso, Chiara. **AutoMATitse: Multiple Aspect Trajectory Data Mining Tool Library.** 2022. 23rd IEEE International Conference on Mobile Data Management (MDM), 2022, pp. xxx-xxx, doi: xxx.

[Bibtex](citation.bib):

```bash
@misc{Portela2022automatise,
    title={},
    author={},
    year={2022},
}
```

### Collaborate with us

Any contribution is welcome. This is an active project and if you would like to include your algorithm in `automatise`, feel free to fork the project, open an issue and contact us.

Feel free to contribute in any form, such as scientific publications referencing `automatise`, teaching material and workshop videos.

### Related packages

- [scikit-mobility](https://github.com/scikit-mobility/scikit-mobility): Human trajectory representation and visualizations in Python;
- [geopandas](https://geopandas.org/en/stable/): Library to help work with geospatial data in Python;
- [movingpandas](https://anitagraser.github.io/movingpandas/): Based on `geopandas` for movement data exploration and analysis.
