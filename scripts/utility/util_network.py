#======================================================
# Network Utility Functions
#======================================================
'''
Info:       Utility functions for network analysis.
Version:    2.0
Author:     Young Lee
Created:    Saturday, 13 April 2019
'''
# Import modules
import os
import sys
from datetime import datetime

import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


#------------------------------
# Utility Functions
#------------------------------
# Set title
def set_title(string):
    # Check if string is too long
    string_size = len(string)
    max_length  = 57
    if string_size > max_length:
        print('TITLE TOO LONG')
    else:
        lr_buffer_len   = int((max_length - string_size) / 2)
        full_buffer_len = lr_buffer_len * 2 + string_size
        print('\n')
        print(full_buffer_len * '=')
        print(full_buffer_len * ' ')
        print(lr_buffer_len * ' ' + string + lr_buffer_len * ' ')
        print(full_buffer_len * ' ')
        print(full_buffer_len * '='+'\n\n')

# Get base graph from nodes and edges
def get_base_graph(graph_nodes, edges):
    try:
        G.clear()
    except NameError:
        pass
    BaseGraph = nx.Graph()
    BaseGraph.add_nodes_from(graph_nodes)
    BaseGraph.add_edges_from(edges)
    return BaseGraph

# Partition graph into communities by Louvain community detection
def get_partition(BaseGraph):
    # Get best partition
    partition = community.best_partition(BaseGraph)
    set_title('Modularity: ' + str(community.modularity(partition, BaseGraph)))
    # Number of communities
    set_title('Number of communities: ' + str(len(np.unique(list(partition.values())))))
    return partition

# Plot graph
def plot_communities(BaseGraph, partition, node_size_df, min_n_nodes_in_comm=1, verbose=0, node_size_scale=0.1, figsize=(16,9)):
    # Keep only communities with a minimum of asins
    centers = {}
    communities = {}
    G_main_com = BaseGraph.copy()
    dcs = []
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        if len(list_nodes) < min_n_nodes_in_comm:
            G_main_com.remove_nodes_from(list_nodes)
        else:
            # Get center
            H = G_main_com.subgraph(list_nodes)
            try:
                d_c = nx.degree_centrality(H)
            except ZeroDivisionError:
                d_c = {list_nodes[0]: 1}
            dcs.append(d_c)
            center = max(d_c, key=d_c.get)
            centers[center] = com
            communities[com] = center
            if verbose:
                # Print community
                print('Community of ', center , '(ID ', com, ') - ', len(list_nodes), ' ASINs:')
                print(list_nodes, '\n')

    # Number of communities filtered out
    set_title('Number of communities after filtering: '+ str(len(communities.keys())) + ' / ' + str(len(np.unique(list(partition.values())))))

    # Display community network
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G_main_com)
    colors = dict(zip(communities.keys(), sns.color_palette('hls', len(communities.keys()))))
    for com in communities.keys():
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        node_sizes = [node_size_df[node_size_df['asin'] == node]['counts'].values[0] * node_size_scale for node in list_nodes]
        nx.draw_networkx_nodes(G_main_com, pos, list_nodes, node_size = node_sizes, node_color = colors[com], alpha=0.8)
    nx.draw_networkx_edges(G_main_com, pos, alpha=0.5)
    centre_labels = {k: k for k,v in centers.items()}
    labels = {k: str(k) for k in pos.keys() if k not in centre_labels.keys()}
    if verbose:
        nx.draw_networkx_labels(G_main_com, pos, labels, font_color='blue', font_size=9)
        nx.draw_networkx_labels(G_main_com, pos, centre_labels, font_color=centre_label_colors, font_size=13,font_weight='bold')
    else:
        for i, com in enumerate(communities.keys()):
            nx.draw_networkx_labels(G_main_com, pos, {k: str(com) + ': '+str(k) for k in [list(centre_labels.values())[i]]}, font_color=colors[com], font_size=13,font_weight='bold')
    plt.axis('off')
    plt.show()

