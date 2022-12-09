#!/usr/bin/env python
'''
Generate autograder test data for Homework3
'''

import copy
import pickle
import sys

import numpy as np
import typer

import me570_graph
from graph_functions import (graph_search_start_goal_vector,
                             graph_search_vector, serialize_save_data_outputs,
                             sphereworldgraph_vector)

sys.path.insert(0, '../common')

import test_data

app = typer.Typer()
DIR_AUTOGRADER = '../../pythonAutograders/homework4'

graph_vector = me570_graph.graph_test_data_load('graphVectorMedium')
nb_vertices = len(graph_vector)
graph = me570_graph.Graph(graph_vector)


@app.command()
def graph_heuristic():
    '''
    Graph.heuristic on random points
    '''

    inputs = [[np.random.randint(nb_vertices),
               np.random.randint(nb_vertices)] for count in range(10)]
    inputs.append([0, 0])
    test_data.generate_save_pairs_object(DIR_AUTOGRADER,
                                         graph,
                                         me570_graph.Graph.heuristic,
                                         inputs,
                                         file_name='Graph.heuristic')


@app.command()
def graph_get_expand_list():
    '''
    Graph.get_expand_list on selected inputs
    '''

    test_idx = [0, 5, 10, 14]
    test_closed = [[],
                   list(range(0, 5)),
                   list(range(0, 10)) + [13],
                   list(range(0, 14))]

    inputs = []
    for idx in test_idx:
        for idx_closed in test_closed:
            inputs.append([idx, idx_closed])

    test_data.generate_save_pairs_object(DIR_AUTOGRADER,
                                         graph,
                                         me570_graph.Graph.get_expand_list,
                                         inputs,
                                         file_name='Graph.get_expand_list')


@app.command()
def graph_expand_element_data():
    '''
    Generate data for testing Graph.expand_element on a sequence of data from sample problems
    '''

    start_goal_pairs = [(3, 11), (0, 14), (14, 0)]
    for idx_pair, (idx_start, idx_goal) in enumerate(start_goal_pairs):
        graph_clone = copy.deepcopy(graph)
        filename = f'graph.graph_expand_element_pair{idx_pair}_exp%d.pkl'
        graph_clone.search(idx_start, idx_goal, save_expansions=filename)


@app.command()
def graph_expand_element():
    '''
    Graph.expand_element on a sequence of data from sample problems, test Graph.graph_vector
    '''

    inout_pairs_vector = []
    inout_pairs_queue = []

    for idx_pair in range(3):
        for idx_expansion in range(12):
            filename = f'graph.graph_expand_element_pair{idx_pair}_exp{idx_expansion}.pkl'
            with open(filename, 'rb') as fin:
                save_data = pickle.load(fin)
                inputs = save_data['inputs']
                outputs = serialize_save_data_outputs(save_data)
                inout_pairs_vector.append((inputs, outputs[0]))
                inout_pairs_queue.append((inputs, outputs[1]))
    test_data.save_pairs(DIR_AUTOGRADER, 'Graph.expand_element_vector',
                         inout_pairs_vector)
    test_data.save_pairs(DIR_AUTOGRADER, 'Graph.expand_element_queue',
                         inout_pairs_queue)


@app.command()
def graph_path():
    '''
    Generate data for testing Graph.path from an already solved graph_vector
    '''

    start_goal_pairs = [(np.random.randint(15), np.random.randint(15))
                        for _ in range(5)]
    start_goal_pairs.append((0, 0))
    inputs = []
    for idx_start, idx_goal in start_goal_pairs:
        graph_clone = copy.deepcopy(graph)
        graph_clone.search(idx_start, idx_goal)
        inputs.append((graph_clone, idx_start, idx_goal))
    test_data.generate_save_pairs(DIR_AUTOGRADER,
                                  me570_graph.Graph.path,
                                  inputs,
                                  file_name='Graph.path')


@app.command()
def graph_search():
    '''
    Generate data for testing Graph.search
    '''

    start_goal_pairs = [(np.random.randint(15), np.random.randint(15))
                        for _ in range(5)]
    start_goal_pairs.append((0, 0))
    inputs = []
    for idx_start, idx_goal in start_goal_pairs:
        graph_clone = copy.deepcopy(graph)
        inputs.append((graph_clone, idx_start, idx_goal))

    test_data.generate_save_pairs(DIR_AUTOGRADER,
                                  graph_search_vector,
                                  inputs,
                                  file_name='Graph.search_vector')

    test_data.generate_save_pairs(DIR_AUTOGRADER,
                                  me570_graph.Graph.search,
                                  inputs,
                                  file_name='Graph.search')


@app.command()
def graph_search_start_goal():
    '''
    Generate data for testing Graph.search_start_goal
    '''

    start_goal_pairs = [(np.random.randint(15), np.random.randint(15))
                        for _ in range(5)]
    start_goal_pairs.append((0, 0))
    inputs = []
    for idx_start, idx_goal in start_goal_pairs:
        graph_clone = copy.deepcopy(graph)
        x_start = graph_clone.graph_vector[idx_start]['x']
        x_goal = graph_clone.graph_vector[idx_goal]['x']
        inputs.append((graph_clone, x_start, x_goal))

    test_data.generate_save_pairs(DIR_AUTOGRADER,
                                  graph_search_start_goal_vector,
                                  inputs,
                                  file_name='Graph.search_start_goal_vector')

    test_data.generate_save_pairs(DIR_AUTOGRADER,
                                  me570_graph.Graph.search_start_goal,
                                  inputs,
                                  file_name='Graph.search_start_goal')


@app.command()
def sphereworldgraph():
    '''
    SphereWorldGraph.__init__ for three sizes
    '''
    inputs = [[5], [10], [15]]
    test_data.generate_save_pairs(DIR_AUTOGRADER,
                                  sphereworldgraph_vector,
                                  inputs,
                                  file_name='SphereWorldGraph')


if __name__ == '__main__':
    app()
