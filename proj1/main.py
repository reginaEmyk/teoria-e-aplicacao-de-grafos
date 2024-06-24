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
import os

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

def question5(graph): 
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
def graph_to_NodeClustering_by_dept(graph_email, df_email_vertices):
  # Assign each node to its department based on df_email_vertices
  communities = {node: dept for node, dept in zip(graph_email.nodes(), df_email_vertices['dept'])}
  # Convert communities dictionary to a list of communities
  community_list = [[] for _ in set(communities.values())]  # Unique department values
  for node, community in communities.items():
    community_list[community].append(node)
  # return clusters defined by vertex department
  return NodeClustering(community_list, graph_email, method_name="Department", overlap=False)

# reference https://cdlib.readthedocs.io/en/latest/reference/evaluation.html
def question6(graph_email, louvain_email, df_email_vertices): # 
    print('Question 6: Compare the modularity of the Louvain community structure with that of the ground truth department structure.')
    ground_truth_communities  = graph_to_NodeClustering_by_dept(graph_email, df_email_vertices)  # clusters defined by vertex department
    ground_truth_modularity = evaluation.modularity_density(graph_email , ground_truth_communities).score
    louvain_modularity = evaluation.modularity_density(graph_email , louvain_email).score
    louvain_modularity_newman_girvan_modularity = louvain_email.newman_girvan_modularity().score
    ground_truth_modularity_newman_girvan_modularity = evaluation.newman_girvan_modularity(graph_email, ground_truth_communities).score
    print('Newman Girvan modularity score of Louvain: ' + str(louvain_modularity_newman_girvan_modularity))
    print('Newman Girvan modularity score of ground truth: ' + str(ground_truth_modularity_newman_girvan_modularity))
    if ground_truth_modularity_newman_girvan_modularity != 0 and louvain_modularity_newman_girvan_modularity:
        print(' Newman Girvan Modularity score percentage of  louvain/ground truth : ', louvain_modularity_newman_girvan_modularity / ground_truth_modularity_newman_girvan_modularity * 100)  
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

def question7(und_graph_email, ground_truth_community, louvain_community):
    print('Question 7: Visualize the graph color-coded by the Louvain community, and then visualize the graph separately color-coded by the ground truth department. Compare the visualizations. Can you describe any of the Louvain communities in terms of departments?') 
    # inform clusters per cluster lists
    print('Ground truth department structure clusters: ' + str(len(ground_truth_community.communities)))
    print('Louvain communities: ' + str(len(louvain_community.communities)))
    # visualize graph for both different cluster methods
    viz.plot_network_clusters(und_graph_email, ground_truth_community, top_k=0)
    plt.title('Ground truth department structure')
    plt.show()
    plt.close()
    viz.plot_network_clusters(und_graph_email, louvain_community, top_k=0)
    plt.title('Louvain communities')
    plt.show()
    print()
    print('Question 7 done.')
    return


# create dataframe with community (int) | vertex id (int) | dept id (int)
def get_dfs_vertex_by_comm_dept(graph, louvain, df_vertices):
    # gets louvain communities into a df
    df_louvain = pd.DataFrame(louvain.communities).transpose()
    # turn vertex to column, community into row. ie vertex | community where vertex is the vertex id and community is the community id
    df_louvain_melted = df_louvain.melt(var_name='community', value_name='vertex')
    # drop rows with NaN values
    df_louvain_melted.dropna(subset=['vertex'], inplace=True)
    # Create a new DataFrame by merging email_vertices with df_louvain_melted
    df_comm_dept_per_vertex = pd.merge(df_louvain_melted, df_vertices, left_on='vertex', right_on='id', how='left')
    # drop repeated column
    df_comm_dept_per_vertex.drop(columns=['id'], inplace=True)
    # print( 'Does the amount of node in graph match the amount of rows in dataframe? ' + str(len(list(graph.nodes())) == len(df_comm_dept_per_vertex))) #  this has been tested and is true but leaving here in case needed
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
    print()
    print('vertex percentage by community:')
    print('community  |  vertex id')
    print(percentage_per_community.head(percentage_per_community.size))
    print()
    print('community  |  vertex id  |  dept')
    print(df_comm_dept_per_vertex)
    print()
    # plotting heatmap
    # Convert percentage_per_community Series to a DataFrame with 'community' as index
    df_heatmap = percentage_per_community.reset_index(name='percentage').rename(columns={'index': 'community'})
    plt.figure(figsize=(10, 6))
    plt.title('Percentage of individuals in each community')
    sns.heatmap(df_heatmap[['percentage']], cmap='YlGnBu', annot=True, fmt='.2f', cbar=True)
    plt.xlabel('Vertex Percentage')
    plt.ylabel('Community ID')
    plt.show()
    print('Question 8 done.')
    print()


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
    amount_largest_cliques = len(largest_cliques)
    print('Largest clique size ' + str(max_clique_size))
    print('Amount of largest cliques ' + str(amount_largest_cliques))
    print('Question 9 done.')
    print()


def visualize_nodes_in_largest_cliques(graph, communities, cliques):
    # create dict with labels 
    node = list(graph.nodes)
    node_label_whitelist = get_unique_vertices_in_list_of_cliques(cliques)
    labels = [vertex \
    for vertex in graph.nodes if vertex in node_label_whitelist]
    nodelabels = dict(zip(node, labels))
    # create and order community mappings
    communities = communities.to_node_community_map()
    ordered_communities = []
    for k in node:
        if communities[k]:
            ordered_communities.append(communities[k].pop())
        else:
            pass
    # create color map
    pastel2 = cm.get_cmap('Pastel2', max(ordered_communities) + 1)
    np.random.seed(123)
    # a label for every node gets confusing but is possible
    nx.draw_spring(graph, labels = nodelabels, cmap = pastel2, node_color = ordered_communities, edge_color = "grey")
    plt.title('Graph with Vertexes in largest cliques labeled')
    plt.show()


def question10(df_cliques, graph, largest_cliques, communities):
    print('Question 10: Try to visualize the members of these cliques in the context of the entire graph. What can you conclude?')
    df_cliques['vertex'] = df_cliques['vertex'].astype(int)
    visualize_nodes_in_largest_cliques(graph, communities, largest_cliques)
    print('Question 10 done.')
    print()
    
      
def question11(graph):
    print('Question 11: Use the Leiden community detection algorithm to find a vertex partition with optimal modularity. How many communities does the Leiden algorithm detect?')
    leiden_comms = algorithms.leiden(graph)
    print('Amount of communities detected by cdlib.algorithms.leiden(): ' + str(len(leiden_comms.communities)))
    print('Question 11 done.')
    print()

def question12(graph, leiden, louvain):
    print('Question 12: Compare the Leiden partition modularity to the Louvain partition modularity.')
    # get modularity metrics for both
    leiden_modularity = evaluation.modularity_density(graph , leiden).score
    louvain_modularity = evaluation.modularity_density(graph , louvain).score
    louvain_modularity_newman_girvan_modularity = louvain.newman_girvan_modularity().score
    leiden_modularity_newman_girvan_modularity = leiden.newman_girvan_modularity().score
    # print modularity scores and percentage of leiden/louvain
    print('Newman Girvan modularity score of Louvain: ' + str(louvain_modularity_newman_girvan_modularity))
    print('Newman Girvan modularity score of Leiden: ' + str(leiden_modularity_newman_girvan_modularity))
    if leiden_modularity_newman_girvan_modularity != 0 and louvain_modularity_newman_girvan_modularity:
        print(' Newman Girvan Modularity score percentage of  louvain/Leiden : ', leiden_modularity_newman_girvan_modularity * 100 / louvain_modularity_newman_girvan_modularity)  
    else:
        print('A modularity is 0 so no percentage is calculated.')
    print()
    print( 'Modularity score of Louvain: ' + str(louvain_modularity))
    print('Modularity score of Leiden: ', leiden_modularity)
    if leiden_modularity != 0 and louvain_modularity:
        print('Modularity score percentage of  louvain/Leiden : ', leiden_modularity * 100 / louvain_modularity)
    else:
        print('A modularity is 0 so no percentage is calculated.')
    print('Question 12 done.')
    print()


def get_unique_vertices_in_list_of_cliques(cliques):
# Use set comprehension to collect unique elements from flattened sublists
  return list(set([vertex for sublist in cliques for vertex in sublist]))

def question13(graph_email,  leiden_communities, louvain_communities, batch_test_filename='leiden_vs_louvain_5000.txt'):
    print('Question 13: Try to use visualization or data exploration methods to determine the main differences between the Leiden and Louvain partitions.')
    
   # plot both clusters 
    viz.plot_network_clusters(graph_email, leiden_communities)
    plt.title('Leiden communities')
    plt.show()
    viz.plot_network_clusters(graph_email, louvain_communities)
    plt.title('Louvain communities')
    plt.show()

    # compare modularity scores for a specific case, like in question6()
    leiden_modularity = evaluation.modularity_density(graph_email , leiden_communities).score
    louvain_modularity = evaluation.modularity_density(graph_email , louvain_communities).score
    louvain_modularity_newman_girvan_modularity = louvain_communities.newman_girvan_modularity().score
    leiden_modularity_newman_girvan_modularity = evaluation.newman_girvan_modularity(graph_email, leiden_communities).score
    print('Newman Girvan modularity score of Louvain: ' + str(louvain_modularity_newman_girvan_modularity))
    print('Newman Girvan modularity score of Leiden: ' + str(leiden_modularity_newman_girvan_modularity))
    if leiden_modularity_newman_girvan_modularity != 0 and louvain_modularity_newman_girvan_modularity:
        print('Newman Girvan Modularity score percentage of  louvain/leiden : ', louvain_modularity_newman_girvan_modularity / leiden_modularity_newman_girvan_modularity * 100)  
    else:
        print('A modularity is 0 so no percentage is calculated.')
    print()
    print( 'Modularity score of Louvain: ' + str(louvain_modularity))
    print('Modularity score of Leiden: ', leiden_modularity)
    if leiden_modularity != 0 and louvain_modularity:
        print('Modularity score percentage of  louvain/leiden : ', louvain_modularity / leiden_modularity * 100)
    else:
        print('A modularity is 0 so no percentage is calculated.')

    # compare average modularity scores ratios from batch generating commmunities. provided in leiden_vs_louvain_5000.txt
    if not os.path.exists(batch_test_filename):
        raise FileNotFoundError(f"The file '{batch_test_filename}' does not exist. Check for typos, filename, comment this line in to create a standard test results file, or run function batch_test_leiden_vs_louvain for a custom test.")
        batch_test_leiden_vs_louvain(und_graph_email, 500, 'leiden_vs_louvain_500.txt') # took less than 10mins in a laptop
        
    modularities_density, modularities_newman_girvan = parse_modularity_file(batch_test_filename)
    print('Test cases count:' + str(len(modularities_density)))

    # get average value for leiden*100/louvain modularity scores
    average_density = sum(sublist[2] for sublist in modularities_density) / len(modularities_density)
    # get average value for leiden*100/louvain newman_girvan modularity scores
    average_newman_girvan = sum(sublist[2] for sublist in modularities_newman_girvan) / len(modularities_newman_girvan)
    print("Average leiden/louvain percentage for modularities_density:", average_density)
    print("Average leiden/louvain percentage for modularities_newman_girvan:", average_newman_girvan)

    # percentage of individuals in each community
    df_louvain_vertex_by_comm, louvain_percentage_per_community = get_dfs_vertex_by_comm_dept(graph_email, louvain_communities, df_email_vertices)
    df_leiden_vertex_by_comm, leiden_percentage_per_community = get_dfs_vertex_by_comm_dept(graph_email, leiden_communities, df_email_vertices)
    print('Percentage of individuals in each community for Louvain:')
    print(louvain_percentage_per_community)
    print('Percentage of individuals in each community for Leiden:')
    print(leiden_percentage_per_community)

    print('Question 13 done.')
    print()
    return


def batch_test_leiden_vs_louvain(graph, iterations, filename):
    print('Batch testing leiden vs louvain')

    # Check if the file already exists
    if os.path.exists(filename):
        raise FileExistsError(f"The file '{filename}' already exists and would be overwritten by batch_test_leiden_vs_louvain().")
    
    modularities_density = []
    modularities_newman_girvan = []
    leiden_communities = []
    louvain_communities = []

    for i in range(iterations):
        leiden_community = algorithms.leiden(graph)
        louvain_community = algorithms.louvain(graph)
        
        leiden_modularity_density = evaluation.modularity_density(graph, leiden_community).score
        louvain_modularity_density = evaluation.modularity_density(graph, louvain_community).score
        
        leiden_modularity_newman_girvan = evaluation.newman_girvan_modularity(graph, leiden_community).score
        louvain_modularity_newman_girvan = evaluation.newman_girvan_modularity(graph, louvain_community).score
        
        # Calculate modularity density percentage
        if louvain_modularity_density != 0:
            mod_density_percentage = leiden_modularity_density * 100 / louvain_modularity_density
        else:
            mod_density_percentage = np.nan
        
        # Calculate Newman-Girvan modularity percentage
        if louvain_modularity_newman_girvan != 0:
            mod_newman_girvan_percentage = leiden_modularity_newman_girvan * 100 / louvain_modularity_newman_girvan
        else:
            mod_newman_girvan_percentage = np.nan
        
        modularities_density.append([leiden_modularity_density, louvain_modularity_density, mod_density_percentage])
        modularities_newman_girvan.append([leiden_modularity_newman_girvan, louvain_modularity_newman_girvan, mod_newman_girvan_percentage])
    
    with open(filename, 'w') as file:
        file.write(f"Modularities Density (leiden, louvain, leiden*100/louvan): {modularities_density}\n")
        file.write(f"Modularities Newman-Girvan (leiden, louvain, leiden*100/louvan): {modularities_newman_girvan}\n")

    print('Modularities Density (leiden, louvain, leiden*100/louvan):', modularities_density)
    print('Modularities Newman-Girvan (leiden, louvain, leiden*100/louvan):', modularities_newman_girvan)
    return modularities_density, modularities_newman_girvan

def parse_modularity_file(filename):
    density_modularity = []
    newman_girvan_modularity = []
    leiden_communities = []
    louvain_communities = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check for expected headers
            if "Modularities Density (leiden, louvain, leiden*100/louvan):" in line:
                data_str = line.split(": ")[-1].strip()  # Extract data after colon
                density_modularity = eval(data_str)  # Use eval to convert string to list
            elif "Modularities Newman-Girvan (leiden, louvain, leiden*100/louvan):" in line:
                data_str = line.split(": ")[-1].strip()  # Extract data after colon
                newman_girvan_modularity = eval(data_str)  # Use eval to convert string to list
    return density_modularity, newman_girvan_modularity

# load csv data and turn edges into undirected graph
df_email_edgelist = email_edgelist()
df_email_vertices = email_vertices()
und_graph_email = nx.from_pandas_edgelist(email_edgelist(), source='from', target='to', create_using=nx.Graph)


question4(und_graph_email, df_email_vertices) # returns largest connected component
question5(und_graph_email) # returns louvain communities
louvain_email = algorithms.louvain(und_graph_email)
leiden_email = algorithms.leiden(und_graph_email)
question13(und_graph_email,  leiden_email, louvain_email, 'leiden_vs_louvain_5000.txt')
question6(und_graph_email, louvain_email, df_email_vertices)
ground_truth_dept_community = graph_to_NodeClustering_by_dept(und_graph_email, df_email_vertices)  
question7(und_graph_email, ground_truth_dept_community, louvain_email)
question8(und_graph_email, louvain_email, df_email_vertices)
question9(und_graph_email)
# for question10
df_comm_dept_per_vertex, percentage_per_community = get_dfs_vertex_by_comm_dept(und_graph_email, louvain_email, df_email_vertices)
und_graph_email = nx.from_pandas_edgelist(email_edgelist(), source='from', target='to', create_using=nx.Graph)
max_clique_size, largest_cliques = largest_cliques_info(und_graph_email)
question10(df_comm_dept_per_vertex, und_graph_email, largest_cliques, louvain_email)
question11(und_graph_email)
question12(und_graph_email, leiden_email, louvain_email)
question13(und_graph_email, louvain_email, leiden_email)
