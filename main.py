# following https://ona-book.org/community.html
#pip install onadata
from onadata import wikivote,email_edgelist, email_vertices
import pandas as pd
import networkx as nx

from cdlib import algorithms, evaluation, NodeClustering
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import seaborn as sns


# https://python-louvain.readthedocs.io/en/latest/api.html
# pip install community python-louvain as networkx.community didnt work
import community as community_louvain
# load csv data 
df_wikivote = wikivote()
df_email_edgelist = email_edgelist()
df_email_vertices = email_vertices()
und_graph_email = nx.from_pandas_edgelist(df_email_edgelist, source='from', target='to', create_using=nx.Graph)

# create directed graph
dir_graph_wikivote = nx.from_pandas_edgelist(df_wikivote, source='from', target='to', create_using=nx.DiGraph())

def question1():
    print('Question 1: Determine how many weakly connected components there are in this graph. How large is the largest component?')
    print(str(nx.number_weakly_connected_components(dir_graph_wikivote)) + ' weakly connected components')

    components = nx.weakly_connected_components(dir_graph_wikivote)
    subgraphs = [dir_graph_wikivote.subgraph(component).copy() 
    for component in components]
    # size of subgraphs
    weakly_components_lengths = [len(subgraph.nodes) for subgraph in subgraphs]
    print('Largest weakly component length: ' + str(max(weakly_components_lengths)))
    print()


def question2():
    print('Question 2: Determine how many strongly connected components there are in this graph. How large is the largest component?')
    #todo mb explain this
    print(str(nx.number_strongly_connected_components(dir_graph_wikivote)) + ' strongly connected components')
#todo mb comment more around here, after doing a report
    components = nx.strongly_connected_components(dir_graph_wikivote)
    subgraphs = [dir_graph_wikivote.subgraph(component).copy() 
    for component in components]
    # size of subgraphs
    strongly_components_lengths = [len(subgraph.nodes) for subgraph in subgraphs]
    print('Largest strongly component length: ' + str(max(strongly_components_lengths)))
    print()

def question3(): # todo 
    print('')
    

def question4(und_graph_email): # hmm.... mb all the connected vertexes r in the same component & there are 1004-986 isolated vertexes
    print('Question 4: Determine the connected components of this network and reduce the network to its largest connected component.')
 #und_graph_email_vertices = nx.from_pandas_edgelist(df_email_vertices, source='from', target='to', create_using=nx.Graph)
# todo reduce und_graph_email_vertices as well
    # get all connected components
    components = list(nx.connected_components(und_graph_email))
    subgraphs = [und_graph_email.subgraph(component).copy() 
    for component in components]
    print('Number of connected components: ' + str(len(components)))

    # size of subgraphs
    components_lengths = [len(subgraph.nodes) for subgraph in subgraphs]
 
# ref https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
    # reduce network to largest component
    largest_cc = max(nx.connected_components(und_graph_email), key=len)
    largest_cc_graph = und_graph_email.subgraph(largest_cc)
 
    print("Size of largest connected component: " + str(len(largest_cc_graph.nodes()))) # TODO !FIX should reduce network to largest connected component
    print()
    global und_graph_email_largest_connected
    global df_email_vertices_largest_connected
    und_graph_email_largest_connected = largest_cc_graph
    # todo do same for edges
    df_email_vertices_largest_connected = df_email_vertices[df_email_vertices['id'].isin([und_graph_email_largest_connected.nodes()])]

    return largest_cc_graph

def question5(graph): # todo
    print('Use the Louvain algorithm to determine a vertex partition/community structure with optimal modularity in this network.')
    
    # create undirected network
    email = nx.from_pandas_edgelist(df_email_edgelist, source = "from", target = "to")
    # get louvain partition which optimizes modularity
    louvain_comms_email = algorithms.louvain(email)
    print(pd.DataFrame(louvain_comms_email.communities).transpose())
    #print('Modularity ' + str(louvain_comms_email.newman_girvan_modularity().score))

# todo ? arent there many louvains? why isnt a list returned? dont i have to pick ? assuming no for now
    return louvain_comms_email

import pandas as pd

#for the ground truth department structure
def graph_to_NodeClustering_obj(graph):
    # reference https://cdlib.readthedocs.io/en/latest/reference/classes.html  
    return NodeClustering([list(graph.nodes)], graph=graph)   


def question6(louvain_email): # 
    print('Question 6: Compare the modularity of the Louvain community structure with that of the ground truth department structure.')
    # todo fix should make community from sharing a dept?
    print('Modularity of louvain in emails edgelist' + str(louvain_email.newman_girvan_modularity().score))
    
    ground_truth_communities  = graph_to_NodeClustering_obj(und_graph_email_largest_connected)  # df_column_to_NodeClustering_obj(und_graph_email, largest_email_vertices, 'dept')
    # reference https://cdlib.readthedocs.io/en/latest/reference/eval/cdlib.evaluation.newman_girvan_modularity.html
    ground_truth_modularity = evaluation.newman_girvan_modularity(und_graph_email_largest_connected , ground_truth_communities).score
    print('Modularity of ground truth department structure in email vertices:', ground_truth_modularity)
    return ground_truth_communities

def visualize_louvain(graph, louvain_comms):
    # Create dict with labels only for Mr Hi and John A
    node_labels = {node: node if node == "Mr Hi" or node == "John A" else "" for node in graph.nodes}

    # Create and order community mappings
    communities = louvain_comms.to_node_community_map()
  # get wont cause error if communities[node] is none 
    communities = [communities[node].get(node) for node in graph.nodes]

    # Create color map
    pastel2 = cm.get_cmap('Pastel2', max(communities) + 1)

    # Visualize Louvain community structure
    plt.figure(figsize=(10, 6))
    np.random.seed(123)
    nx.draw_spring(graph, labels=node_labels, cmap=pastel2, node_color=communities, edge_color="grey")
    plt.title("Graph color-coded by Louvain community")
    plt.show()



def visualize(graph, communities):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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


def question8(graph, louvain, df_vertices):
    print('Question 8: Create a dataframe containing the community and department for each vertex. Manipulate this dataframe to show the percentage of individuals from each department in each community. Try to visualize this using a heatmap or other style of visualization and try to use this to describe the communities in terms of departments.')
    df_louvain = pd.DataFrame(louvain_email.communities).transpose()
    # Melt the DataFrame df_louvain to have 'community' and 'vertex' columns
    df_louvain_melted = df_louvain.melt(var_name='community', value_name='vertex')
    df_louvain_melted.dropna(subset=['vertex'], inplace=True)

    # Create a new DataFrame by merging email_vertices with df_louvain_melted
    df_comm_dept_per_vertex = pd.merge(df_louvain_melted, df_email_vertices, left_on='vertex', right_on='id', how='left')
    # drop repeated column
    df_comm_dept_per_vertex.drop(columns=['id'], inplace=True)
    # Drop rows with NaN values

    print(df_comm_dept_per_vertex) #  986 nodes ok
    # Calculate the total number of rows
    total_rows = df_comm_dept_per_vertex.shape[0]

    # Count the occurrences of each community value
    community_counts = df_comm_dept_per_vertex['community'].value_counts()

    # Calculate the percentage of each community
    percentage_per_community = community_counts / total_rows * 100
    print('vertex percentage by community:')
    print('community  |  vertex id')
    print(percentage_per_community)
    # Call question8 with the required parameters
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert percentage_per_community Series to a DataFrame with 'community' as index
    df_heatmap = percentage_per_community.reset_index(name='percentage').rename(columns={'index': 'community'})

    # Plotting the heatmap
    plt.figure(figsize=(10, 6))
    plt.title('Percentage of individuals in each community')
    sns.heatmap(df_heatmap[['percentage']], cmap='YlGnBu', annot=True, fmt='.2f', cbar=True)
    plt.xlabel('Vertex Percentage')
    plt.ylabel('Community ID')
    plt.show()

# ref https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.clique.find_cliques.html
def question9(graph):
    print('Find the largest clique size in the graph. How many such largest cliques are there? What do you think a clique represents in this context?')
    clique_sizes = [len(c) for c in sorted(nx.find_cliques(graph), key=len, reverse=True)]
    max_clique_size = max(clique_sizes)

    amount_largest_cliques = clique_sizes.count(max_clique_size)
    print('Largest clique size ' + str(max_clique_size))
    print('Amount of largest cliques ' + str(amount_largest_cliques))
    return clique_sizes

# question1()
# question2()
# question3()

df_email_edgelist = email_edgelist()
und_graph_email = nx.from_pandas_edgelist(df_email_edgelist, source='from', target='to', create_using=nx.Graph)
#und_graph_email = question4(und_graph_email) # network reduced to largest
question4(und_graph_email)

louvain_email = question5(und_graph_email)
print(louvain_email)
# ground_truth_dept_community = question6(louvain_email)

# question7(ground_truth_dept_community) # todo fix looks sketchy that theyre all the same color-- all same dept?

# question8(und_graph_email, louvain_email, df_email_vertices)
print(len(question9(und_graph_email)))

# checking amount of nodes that should be in graph
# print(nx.from_pandas_edgelist(email_edgelist(), source='from', target='to', create_using=nx.Graph).number_of_nodes())


# pd.DataFrame(louvain_comms.communities).transpose()