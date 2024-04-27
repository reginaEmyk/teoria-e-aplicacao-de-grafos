# following https://ona-book.org/community.html
#pip install onadata
from onadata import wikivote
import pandas as pd
import networkx as nx

# load csv data 
df_wikivote = wikivote()

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
    

    
question1()
question2()
question3()