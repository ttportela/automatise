# -*- coding: utf-8 -*-
'''
Multiple Aspect Trajectory Data Mining Tool Library

The present application offers a tool, to support the user in the classification task of multiple aspect trajectories, specifically for extracting and visualizing the movelets, the parts of the trajectory that better discriminate a class. It integrates into a unique platform the fragmented approaches available for multiple aspects trajectories and in general for multidimensional sequence classification into a unique web-based and python library system. Offers both movelets visualization and a complete configuration of classification experimental settings.

Created on Dec, 2021
Copyright (C) 2022, License GPL Version 3 or superior (see LICENSE file)

@author: Tarlis Portela
'''
PACKAGE_NAME = 'automatise'
DATA_PATH = './app/datasets'
EXP_PATH  = '/app/results'
README    = PACKAGE_NAME+'/README.md'

PAGES_ROUTE =  PACKAGE_NAME+'/assets'

RESULTS_FILE    = PACKAGE_NAME+'/assets/experimental_history.csv'

# page_title = 'Tarlis\'s Multiple Aspect Trajectory Analysis'
page_title = PACKAGE_NAME.capitalize()

def underDev(pathname):
    import dash_bootstrap_components as dbc
    from dash import html
    return html.Div([
            dbc.Alert('Page "{}" not found, sorry. ;/'.format(pathname), color="info", style = {'margin':10})
        ])

def alert(msg, mtype="info"):
    import dash_bootstrap_components as dbc
    return dbc.Alert(msg, color=mtype, style = {'margin':10})

def render_markdown_file(file, div=False):
    from dash import html
    from dash import dcc
    f = open(file, "r")
    if div:
        return html.Div(dcc.Markdown(f.read()), style={'margin': '20px'}, className='markdown')
    else:
        return dcc.Markdown(f.read())