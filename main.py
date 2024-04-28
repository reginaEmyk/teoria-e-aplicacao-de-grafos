# following https://ona-book.org/community.html
# reference: https://python-louvain.readthedocs.io/en/latest/api.html
# pip install onadata igraph leidenalg python-louvain
# pip install community python-louvain as networkx.community didnt work
from onadata import email_edgelist, email_vertices
import pandas as pd
import networkx as nx
from cdlib import algorithms, evaluation, NodeClustering, viz
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import seaborn as sns
import community as community_louvain
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def question4(und_graph_email, df_email_vertices):
    print('Question 4: Determine the connected components of this network and reduce the network to its largest connected component.')
    # get all connected components
    components = list(nx.connected_components(und_graph_email))
    # copy each component to a new graph
    subgraphs = [und_graph_email.subgraph(component).copy() 
    for component in components]
    print('Number of connected components: ' + str(len(components)))
    # size of subgraphs
    components_lengths = [len(subgraph.nodes) for subgraph in subgraphs] 
    print('Sizes of connected components: ' + str(components_lengths))

    # a simple way to get largest component
    # ref https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
    largest_cc = max(nx.connected_components(und_graph_email), key=len)
    largest_cc_graph = und_graph_email.subgraph(largest_cc)
    print("Size of largest connected component: " + str(len(largest_cc_graph.nodes())))
    
    
    # get df for vertices & df for edges in largest connected component
    # update df_email_vertices to only include vertices in largest connected component. note that email_vertices() has department info not present in graph
    df_email_vertices_largest_connected = df_email_vertices[df_email_vertices['id'].isin(list(largest_cc_graph.nodes()))]
    # get a df of edges in largest component, same structure as email_edgelist()
    df_edges_largest_connected = pd.DataFrame(list(largest_cc_graph.edges()), columns=['from', 'to'])
    print('Largest connected component vertices (first 3 rows of df):')
    print(df_email_vertices_largest_connected.head(3))
    print('Largest connected component edges (first 3 rows of df):')
    print(df_edges_largest_connected.head(3))
    print('Original Network:')
    print(und_graph_email)
    print('Reduced Network:')
    print(largest_cc_graph)
    print('Question 4 done.')
    print()
    return largest_cc_graph, df_email_vertices_largest_connected, df_edges_largest_connected

def question5(graph): # todo
    print('Question 5: Use the Louvain algorithm to determine a vertex partition/community structure with optimal modularity in this network.')    
    # get louvain partition which optimizes modularity
    louvain_comms_email = algorithms.louvain(graph)
    print('Louvain communities in email edgelist:')
    print(pd.DataFrame(louvain_comms_email.communities).transpose())
    #print(pd.DataFrame(louvain_comms_email.communities).dtypes)
    print('Question 5 done.')
    print()
    return louvain_comms_email


#for the ground truth department structure
def graph_to_NodeClustering_obj(graph):
    # reference https://cdlib.readthedocs.io/en/latest/reference/classes.html  
    return NodeClustering([list(graph.nodes)], graph=graph)   

# reference https://cdlib.readthedocs.io/en/latest/reference/evaluation.html
def question6(graph_email, louvain_email): # 
    print('Question 6: Compare the modularity of the Louvain community structure with that of the ground truth department structure.')
    ground_truth_communities  = graph_to_NodeClustering_obj(graph_email)  # turns whole graph into a node cluster
    ground_truth_modularity = evaluation.modularity_density(graph_email , ground_truth_communities).score
    louvain_modularity = evaluation.modularity_density(graph_email , louvain_email).score
    louvain_modularity_newman_girvan_modularity = louvain_email.newman_girvan_modularity().score
    ground_truth_modularity_newman_girvan_modularity = evaluation.newman_girvan_modularity(graph_email, ground_truth_communities).score
    louvain_email_modularity_overlap = evaluation.modularity_overlap(graph_email , louvain_email).score
    ground_truth_modularity_modularity_overlap = evaluation.modularity_overlap(graph_email , ground_truth_communities).score
    print('Newman Girvan modularity score of Louvain: ' + str(louvain_modularity_newman_girvan_modularity))    
    print('Newman Girvan modularity score of ground truth: ' + str(ground_truth_modularity_newman_girvan_modularity))
    if ground_truth_modularity_newman_girvan_modularity != 0 and louvain_modularity_newman_girvan_modularity:
        print(' Newman Girvan Modularity score percentage of  louvain/ground truth : ', louvain_modularity_newman_girvan_modularity / ground_truth_modularity_newman_girvan_modularity * 100)  
    else:
        print('A modularity is 0 so no percentage is calculated.')
    print()
    print( 'Modularity of overlapping communities score of Louvain: ' + str(louvain_email_modularity_overlap))
    print( 'Modularity of overlapping communities score of ground truth: ' + str(ground_truth_modularity_modularity_overlap))
    if ground_truth_modularity_modularity_overlap != 0 and louvain_email_modularity_overlap:
        print('Modularity score percentage of  louvain/ground truth : ', louvain_email_modularity_overlap / ground_truth_modularity_modularity_overlap * 100)
    else:
        print('A modularity is 0 so no percentage is calculated.')
    print()
    print( 'Modularity score of Louvain: ' + str(louvain_modularity))
    print('Modularity score of ground truth: ', ground_truth_modularity)
    if ground_truth_modularity != 0 and louvain_modularity:
        print('Modularity score percentage of  louvain/ground truth : ', louvain_modularity / ground_truth_modularity * 100)
    else:
        print('A modularity is 0 so no percentage is calculated.')
    print()
    print('Question 6 done.')
    return 

def visualize(graph, communities):
    # create dict with labels 
    node = list(graph.nodes)
    labels = [i \
    for i in graph.nodes]
    nodelabels = dict(zip(node, labels))
# todo add legend per dept
    # create and order community mappings
    communities = communities.to_node_community_map()
    # communities = [communities[k].pop() for k in node]
    # due to previous error: network reduced to a single node or not reduced, attempted pop from empty list
    ordered_communities = []
    for k in node:
        if communities[k]:
            ordered_communities.append(communities[k].pop())
        else:
            # If communities[k] is empty, keep it as it is
            pass
            # ordered_communities.append(None) # Filter out None values
    # create color map
    pastel2 = cm.get_cmap('Pastel2', max(ordered_communities) + 1)
    # visualize
    np.random.seed(123)
    # todo add nodelabels as legend
    nx.draw_spring(graph, labels = nodelabels, cmap = pastel2, node_color = ordered_communities, edge_color = "grey")
    # nx.draw_spring(graph, cmap = pastel2, node_color = ordered_communities, edge_color = "grey")

    plt.show()

def question7(ground_truth_community):
    print('Question 7: Visualize the graph color-coded by the Louvain community, and then visualize the graph separately color-coded by the ground truth department. Compare the visualizations. Can you describe any of the Louvain communities in terms of departments?') 
    print('Louvain community structure')
    # Visualize Louvain community structure
    print('Ground truth department structure')
    print(ground_truth_community)
    visualize(und_graph_email, louvain_email)
    # todo this makes no sense it has a single dept
    visualize(und_graph_email, ground_truth_community)


def get_dfs_vertex_by_comm_dept(graph, louvain, df_vertices):
    df_louvain = pd.DataFrame(louvain.communities).transpose()
    # Melt the DataFrame df_louvain to have 'community' and 'vertex' columns
    df_louvain_melted = df_louvain.melt(var_name='community', value_name='vertex')
    df_louvain_melted.dropna(subset=['vertex'], inplace=True)
    # Create a new DataFrame by merging email_vertices with df_louvain_melted
    df_comm_dept_per_vertex = pd.merge(df_louvain_melted, df_vertices, left_on='vertex', right_on='id', how='left')
    # drop repeated column
    df_comm_dept_per_vertex.drop(columns=['id'], inplace=True)
    # Drop rows with NaN values
    print(df_comm_dept_per_vertex) #  986 nodes ok
    # get the total number of rows
    total_rows = df_comm_dept_per_vertex.shape[0]
    # Count the occurrences of each community value
    community_counts = df_comm_dept_per_vertex['community'].value_counts()
    # Calculate the percentage of each community
    percentage_per_community = community_counts / total_rows * 100
    return df_comm_dept_per_vertex, percentage_per_community

def question8(graph, louvain, df_vertices):
    print('Question 8: Create a dataframe containing the community and department for each vertex. Manipulate this dataframe to show the percentage of individuals from each department in each community. Try to visualize this using a heatmap or other style of visualization and try to use this to describe the communities in terms of departments.')    
    df_comm_dept_per_vertex, percentage_per_community = get_dfs_vertex_by_comm_dept(graph, louvain, df_vertices)   
    print('vertex percentage by community:')
    print('community  |  vertex id')
    print(percentage_per_community)
    print('community  |  vertex id  |  dept')
    print(df_comm_dept_per_vertex)
    # plotting heatmap
    # Convert percentage_per_community Series to a DataFrame with 'community' as index
    df_heatmap = percentage_per_community.reset_index(name='percentage').rename(columns={'index': 'community'})
    plt.figure(figsize=(10, 6))
    plt.title('Percentage of individuals in each community')
    sns.heatmap(df_heatmap[['percentage']], cmap='YlGnBu', annot=True, fmt='.2f', cbar=True)
    plt.xlabel('Vertex Percentage')
    plt.ylabel('Community ID')
    plt.show()

# returns lagrest clique size, list of largest cliques
def largest_cliques_info(graph):
    cliques = sorted(nx.find_cliques(graph), key=len, reverse=True)
    clique_sizes = [len(c) for c in cliques]
    max_clique_size = max(clique_sizes)
    largest_cliques = [c for c in cliques if len(c) == max_clique_size]
    return max_clique_size, largest_cliques

# ref https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html
def question9(graph):
    print('Question 9: Find the largest clique size in the graph. How many such largest cliques are there? What do you think a clique represents in this context?')
    max_clique_size, largest_cliques = largest_cliques_info(graph)
    amount_largest_cliques = largest_cliques.count(max_clique_size)
    print('Largest clique size ' + str(max_clique_size))
    print('Amount of largest cliques ' + str(amount_largest_cliques))
    

def question10(df_cliques, largest_cliques):
    print('Question 10: Try to visualize the members of these cliques in the context of the entire graph. What can you conclude?')
    df_cliques['vertex'] = df_cliques['vertex'].astype(int)
    # Iterate through cliques in largest_cliques
    for i, clique in enumerate(largest_cliques):
        # Create a new column name based on the clique index
        column_name = f"clique{i}"
        # Define a lambda function to check if vertex is in the clique
        df_cliques[column_name] = df_cliques['vertex'].isin(clique)
        # # print(len(cliques))
        # print((largest_cliques))
        print(df_cliques.head(7))
      
def question11(graph):
    print('Question 11: Use the Leiden community detection algorithm to find a vertex partition with optimal modularity. How many communities does the Leiden algorithm detect?')
    leiden_comms = algorithms.leiden(graph)
    print('Amount of communities detected by cdlib.algorithms.leiden(): ' + str(len(leiden_comms.communities)))

def question12(graph, leiden, louvain):
    print('Question 12: Compare the Leiden partition modularity to the Louvain partition modularity.')

    leiden_modularity = evaluation.newman_girvan_modularity(graph , leiden).score
    louvain_modularity = evaluation.newman_girvan_modularity(graph , louvain).score
    print('Leiden modularity: ' + str(leiden_modularity))
    print('Louvain modularity: ' + str(louvain_modularity))
    print('Leiden/Louvain percentage: ' + str((leiden_modularity / louvain_modularity) * 100))


def get_unique_vertices_in_list_of_cliques(cliques):
# Use set comprehension to collect unique elements from flattened sublists
  return list(set([vertex for sublist in cliques for vertex in sublist]))


def compare_community_algos(graph, communities_list):
    for community_algo_results in communities_list:
        viz.plot_network_clusters(graph, community_algo_results)
    plt.show()
    

def question13(graph, communities_algos_result_list):
    compare_community_algos(graph, communities_algos_result_list)



# load csv data and turn edges into undirected graph
df_email_edgelist = email_edgelist()
df_email_vertices = email_vertices()
und_graph_email = nx.from_pandas_edgelist(df_email_edgelist, source='from', target='to', create_using=nx.Graph)


graph = nx.from_pandas_edgelist(df_email_edgelist, source='from', target='to', create_using=nx.Graph)
# question4(und_graph_email, df_email_vertices) # returns largest connected component
# question5(und_graph_email) # returns louvain communities

louvain_email = algorithms.louvain(graph)
question6(graph, louvain_email)

# question7(ground_truth_dept_community) # todo fix looks sketchy that theyre all the same color-- all same dept?
# question8(und_graph_email, louvain_email, df_email_vertices)
# df_comm_dept_per_vertex, percentage_per_community = get_dfs_vertex_by_comm_dept(und_graph_email, louvain_email, df_email_vertices)
# # question9(und_graph_email)
# max_clique_size, largest_cliques = largest_cliques_info(und_graph_email)
# vertices_in_largest_cliques = get_unique_vertices_in_list_of_cliques(largest_cliques)

# df_cliques = df_comm_dept_per_vertex.copy()
# # question10(df_cliques, largest_cliques)

# # checking amount of nodes that should be in graph
# # print(nx.from_pandas_edgelist(email_edgelist(), source='from', target='to', create_using=nx.Graph).number_of_nodes())


# # question11(und_graph_email)
# leiden_email = algorithms.leiden(und_graph_email)
# question13(und_graph_email, [louvain_email, leiden_email])

# COMPARE REF https://cdlib.readthedocs.io/en/latest/reference/viz.html



# TODO IN COMPARE COMMUNITES USE MODILARITY SCORES
# TODO VISUAL REPRESENTATION FOR COMPARE QUESTIONS
