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

import base64
import datetime
import io

import dash
from dash import dash_table
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State

import pandas as pd

from automatize.movelets import *

# ------------------------------------------------------------
# attributes = ['None']
# sel_attributes = []
# ------------------------------------------------------------
def page_trajectories_filter(ls_trajs, from_trajs, to_trajs, attributes, sel_attributes):
    return html.Div([
        html.Strong('Range of Trajectories ('+str(len(ls_trajs))+'): '),
        dcc.RangeSlider(
            id='input-range-trajs',
            min=0,
            max=len(ls_trajs) if len(ls_trajs) > 0 else 100,
            value=[from_trajs, to_trajs],
            tooltip={"placement": "bottom", "always_visible": True}
        ),
#         html.Strong('From: '),
#         dcc.Input(
#             id='input-from-traj',
#             placeholder='From...',
#             type='text',
#             value=str(from_trajs)
#         ),
#         html.Strong(' To: '),
#         dcc.Input(
#             id='input-to-traj',
#             placeholder='To...',
#             type='text',
#             value=str(to_trajs)
#         ),
        html.H6('Attributes: '),
        dcc.Dropdown(
            id='input-attr-traj',
            options=[
                {'label': attr, 'value': attr} for attr in attributes
            ],
            multi=True,
            value=",".join(sel_attributes)
        ),
        html.Hr(),
    ], style = {'display':'inline'})
    
def getAttrSize(T, sel_attributes):
    size = 0
    for i in range(T.size):
        for attr in sel_attributes:
            size = max(size, len(str(T.points[i][attr])))
    return size
    
CH_SIZE = 10
def getAttrCHS(length, size):
    return str(length*size*(CH_SIZE/2)+length*(CH_SIZE/2)*3)+'px'

def render_page_trajectories(ls_trajs, range_value, ls_movs, sel_attributes):
    ncor = 7
    ls_components = []
    
    from_trajs, to_trajs = range_value
    
    if len(ls_trajs) > 0:
        attributes = ls_trajs[0].attributes
#         print(sel_attributes)
        if sel_attributes == '':
            sel_attributes = attributes
        else:
            sel_attributes = sel_attributes if isinstance(sel_attributes, list) else sel_attributes.split(',') 
            
#         size = 10 # in chars
#         chsz = 12
        import random
        T = random.sample(ls_trajs, 1)[0]
        size = getAttrSize(T, sel_attributes) # in chars
                
        ls_components.append(page_trajectories_filter(ls_trajs, from_trajs, to_trajs, attributes, sel_attributes))

        ls_trajs = ls_trajs[(from_trajs if from_trajs < len(ls_trajs)-1 else 0) : (to_trajs if to_trajs < len(ls_trajs)-1 else 100)]
        for k in range(len(ls_trajs)):
    #         points = T.points_trans()
            T = ls_trajs[k]
            ls_components.append(html.Div(className='traj-color'+str((k % ncor) + 1)+'-rangeslider traj-slider', children = [
                html.Div(style={'float': 'left', 'textAlign': 'center', 'width': '50px', 'fontSize': str(CH_SIZE)+'px'}, children = [
                    html.Span(T.tid), 
                    html.Br(),
                    html.Strong(T.label),
                ]),
                html.Div(style={'marginLeft': '50px'}, children = [dcc.RangeSlider(
                    marks={i: {'label':'p'+str(i)} for i in range(T.size)}, # , 'style':{'display': 'block'}
                    min=0,
                    max=T.size-1,
                    value=[0, T.size-1],
    #                 disabled=True,
                )]),
            ], style={'width': getAttrCHS(T.size, size)}))
            for attr in T.attributes:
                if attr in sel_attributes:
                    values = []
                    for m in ls_movs:
                        if m.tid == T.tid and attr in m.attributes():
                            values += [m.start, m.start+m.size]
#                     print(values)
                    ls_components.append(html.Div(
                        className='traj-color'+str((k % ncor) + 1)+'-rangeslider traj-slider traj-slider', children = [
                        html.A(attr, style={'float': 'left', 'textAlign': 'center', 'width': '50px', 'fontSize': str(CH_SIZE)+'px'}),
                        html.Div(style={'marginLeft': '50px'}, children = [dcc.RangeSlider(
                            marks={i: str(T.points[i][attr]) for i in range(T.size)}, # , 'style':{'display': 'block'}
                            min=0,
                            max=T.size-1,
#                             value=[0, T.size-1],
                            value=list(set(values)),
                            disabled=True,
                        )]),
                    ], style={'width': getAttrCHS(T.size, size)}))
    #         ls_components.append(dash_table.DataTable(
    #             id='table-tid-'+str(T.tid),
    #             style_data={
    #                 'fontSize': '8px',
    # #                 'whiteSpace': 'normal',
    # #                 'height': 'auto',
    # #                 'width': '100%',
    #             },
    #             style_table={'minWidth': '100%'},
    #             style_cell={
    #                 'overflow': 'hidden',
    #                 'textOverflow': 'ellipsis',
    # #                 'maxWidth': '10%',
    #                 'width': '50px',
    #                 'textAlign': 'center',
    #             },
    #             style_cell_conditional=[
    #                 {'if': {'column_id': 'attr'},
    #                  'minWidth': '50px', 'maxWidth': '50px'},
    #             ],
    #             columns=[{"name": str(T.tid), "id": 'attr'}]+[{"name": 'p'+str(i), "id": 'p'+str(i)} for i in range(T.size)],
    #             data=T.points_trans(),
    #         ))
            ls_components.append(html.Hr())
    else:
        ls_components.append(page_trajectories_filter(ls_trajs, from_trajs, to_trajs, [], sel_attributes))
    
    return html.Div(ls_components)

def render_page_movelets(T, ls_movs):
#     ncor = 7
    ls_components = []
    attributes = T.attributes
    size = getAttrSize(T, attributes)

    ls_components.append(html.Div(children = [
        html.Div(style={'float': 'left', 'textAlign': 'center', 'width': '50px', 'fontSize': str(CH_SIZE)+'px'}, children = [
            html.Span(T.tid), 
            html.Br(),
            html.Strong(T.label),
        ]),
        html.Div(style={'marginLeft': '50px'}, children = [dcc.RangeSlider(
            marks={i: {'label':'p'+str(i)} for i in range(T.size)}, # , 'style':{'display': 'block'}
            min=0,
            max=T.size-1,
            value=[0, T.size-1],
        )]),
    ], style={'width': getAttrCHS(T.size, size)}))
    ls_components.append(html.Hr())
    
#     print(ls_movs)
    for m in ls_movs:
        if m.tid == T.tid:
            ls_components.append(html.H6('Movelet: '+str(m.mid)))
            for attr in m.attributes():
                ls_components.append(html.Div(children = [
                    html.A(attr, 
                           style={'float': 'left', 'textAlign': 'center', 'width': '50px', 'fontSize': str(CH_SIZE)+'px'}),
                    html.Div(style={'marginLeft': '50px'}, children = [dcc.RangeSlider(
                        marks={i: str(T.points[i][attr]) for i in range(T.size)}, # , 'style':{'display': 'block'}
                        min=0,
                        max=T.size-1,
    #                             value=[0, T.size-1],
                        value=[m.start, m.start+m.size],
                        tooltip={"placement": "bottom", "always_visible": False},
        #                 style={'display': 'block'}
    #                         disabled=True,
                    )]),
                ], style={'width': getAttrCHS(T.size, size)}))
            ls_components.append(html.Hr())
    
    return html.Div(ls_components)