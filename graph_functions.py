'''
Helper functions to test outputs of methods from the Graph class
'''

import me570_graph


def unqueue(pq_open):
    '''
    Transforms contents of a PriorityQueue into a list of pairs
    '''
    items = []
    while True:
        key, val = pq_open.min_extract()
        if key is None:
            break
        items.append((key, val))
    items = sorted(items, key=lambda x: x[0])
    return items


def serialize_save_data_outputs(save_data):
    '''
    Transform outputs into a pair (graph_vector, unqueue(pq_open))
    '''
    return (save_data['outputs'][0].graph_vector,
            unqueue(save_data['outputs'][1]))


def graph_search_vector(graph, *args):
    '''
    Wrapper for Graph.search that returns only graph_vector
    '''
    graph.search(*args)
    return graph.graph_vector


def graph_search_start_goal_vector(graph, *args):
    '''
    Wrapper for Graph.search that returns only graph_vector
    '''
    graph.search_start_goal(*args)
    return graph.graph_vector


def sphereworldgraph_vector(nb_cells):
    '''
    Wrapper for Graph.search that returns only graph_vector
    '''
    graph = me570_graph.SphereWorldGraph(nb_cells).graph
    return graph.graph_vector
