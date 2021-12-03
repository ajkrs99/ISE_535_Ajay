#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import seaborn as sns
import numpy as np
from mip import *
import os
import matplotlib.pyplot as plt
import streamlit as st
import time


# In[49]:


df=pd.read_excel("Project_data.xlsx", sheet_name="Plant Data")
df1=pd.read_excel("Project_data.xlsx", sheet_name="DC Data")
df2=pd.read_excel("Project_data.xlsx", sheet_name="Customer Data")


# In[50]:


plant=df['Mfg Plant'].tolist()
plcapacity=df['Capacity'].tolist()
plvarcost= df['Variable Cost'].tolist()
dc=df1['DC'].tolist()
dcfixcost=df1['Fixed Cost'].tolist()
dcvarcost=df1['Variable Cost'].tolist()
dccapacity=df1['Capacity'].tolist()
region= df2['Customer Locations']
demand=df2['Demand'].tolist()


# In[4]:


indist=[]
for i in range(len(plant)):
    a=[]
    for j in range(len(dc)):
        dist=((df['X'][i]-df1['X'][j])**2 + (df['Y'][i]-df1['Y'][j])**2)
        a.append(dist)
    indist.append(a)


# In[5]:


outdist=[]
for i in range(len(dc)):
    a=[]
    for j in range(len(region)):
        dist=((df1['X'][i]-df2['X'][j])**2 + (df1['Y'][i]-df2['Y'][j])**2)
        a.append(dist)
    outdist.append(a)


# In[9]:


numDC=3


# In[10]:


time.sleep(1)


# In[27]:


M=xsum(demand[i] for i in range(len(demand)))


# In[28]:


model = Model()


# In[29]:


inflow= [[model.add_var() for j in range(len(dc))] for i in range(len(plant))]
outflow= [[model.add_var() for k in range(len(region))] for j in range(len(dc))]
opendc=[model.add_var(var_type=BINARY) for i in range(len(dc))]


# In[30]:


model.objective=minimize(xsum(plvarcost[i]*inflow[i][j] for j in range(len(dc)) for i in range(len(plant))) +                         xsum(dcfixcost[j]*opendc[j] for j in range(len(dc))) +                         xsum(dcvarcost[j]*outflow[j][k] for j in range(len(dc)) for k in range(len(region))) +                         xsum(inflow[i][j]*indist[i][j] for i in range(len(plant)) for j in range(len(dc))) +                         xsum(outflow[j][k]*outdist[j][k] for j in range(len(dc)) for k in range(len(region))))


# In[31]:


for i in range(len(plant)):
    model += xsum(inflow[i][j] for j in range(len(dc))) <= plcapacity[i]

for j in range(len(dc)):
    model += xsum(outflow[j][k] for k in range(len(region))) <= dccapacity[j]

for k in range(len(region)):
    model += xsum(outflow[j][k] for j in range(len(dc))) == demand[k]

for j in range(len(dc)):
    model += xsum(inflow[i][j] for i in range(len(plant))) == xsum(outflow[j][k] for k in range(len(region)))
    
for j in range(len(dc)):
    model += xsum(outflow[j][k] for k in range(len(region))) <= M.x*opendc[j]

model += xsum(opendc[j] for j in range(len(dc))) ==3


# In[32]:


model.optimize()


# In[33]:


st.write("The objective Value is:")


# In[34]:


model.objective_value


# In[35]:


result=pd.DataFrame(data=None, index= plant, columns=dc)


# In[36]:


st.write("Units moved from each plant to DCs")


# In[37]:


for i in plant:
    for j in dc:
        result.at[i,j]= inflow[plant.index(i)][dc.index(j)].x
result


# In[38]:


result2= pd.DataFrame(data=None, index=dc, columns=region)


# In[ ]:


st.write("Units moved from each DC to Customers")


# In[ ]:


for j in range(len(dc)): 
    for k in range(len(region)):
       result2.iat[j,k]= outflow[j][k].x
result2 


# In[ ]:


import plotly.graph_objects as go

import networkx as nx

G = nx.Graph()




# In[ ]:


# adding nodes

for cust in region:
    G.add_node(cust, size = 2)

for center in dc:
    G.add_node(center, size = 5)

  
G.add_node(plant[0], size = 5)
G.add_node(plant[1], size = 5)

pos_ = nx.random_layout(G)

# make node traces
# Make a node trace
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "top center",
                        textfont_size = 8,
                        mode      = 'markers+text',
                        hoverinfo = 'none',
                        marker    = dict(color = [],
                                         size  = [],
                                         line  = None))
# For each node in midsummer, get the position and size and add to the node_trace
for node in G.nodes():
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    if node in dc:
        node_trace['marker']['color'] += tuple(['MediumPurple'])
    elif node in plant:
        node_trace['marker']['color'] += tuple(['red'])
    else:
        node_trace['marker']['color'] += tuple(['cornflowerblue'])
    node_trace['marker']['size'] += tuple([5*G.nodes()[node]['size']])
    node_trace['text'] += tuple(['<b>' + node + '</b>'])
    
    
# adding edges
for i in range(len(dc)):
    for cust in region:
        if result2.iloc[i][cust] > 0:
            G.add_edge(dc[i], cust, weight = result2.iloc[i][cust])
            
for i in range(len(plant)):
    for center in dc:
        if result.iloc[i][center] > 0:
            G.add_edge(plant[i], center, weight = result.iloc[i][center])
            

def make_edge(x, y, text, width):
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'cornflowerblue'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')


# For each edge, make an edge_trace, append to list
edge_trace = []
for edge in G.edges():
    if G.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]
        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]
        text   = char_1 + '--' + char_2 + ': ' + str(G.edges()[edge]['weight'])
        trace  = make_edge([x0, x1, None], [y0, y1, None], text, width = 0.5)
        edge_trace.append(trace)
    

        
# display graph
# Create figure
fig = go.Figure()

# Add node trace
fig.add_trace(node_trace)

# Add all edge traces
for trace in edge_trace:
    fig.add_trace(trace)
    
# Remove legend
fig.update_layout(showlegend = False)
# Remove tick labels
fig.update_xaxes(showticklabels = False)
fig.update_yaxes(showticklabels = False)
# Show figure
fig.show()

