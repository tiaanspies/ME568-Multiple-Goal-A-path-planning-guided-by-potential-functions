"""
Functions to test algorithms and create plots
"""

import matplotlib.pyplot as plt
import numpy as np

import me570_graph
import me570_robot


def graph_search_test():
    """
    Call graph_search to find a path between the bottom left node and the
    top right node of the  graphVectorMedium graph from the
    graph_test_data_load function (see Question~ q:graph test data). Then
    use Graph.plot() to visualize the result.
    """

    graph = me570_graph.Graph(
        me570_graph.graph_test_data_load('graphVectorMedium'))
    x_path = graph.search(0, 14, flag_debug_visualization=True)
    plt.close('all')
    graph.plot(flag_labels=False,
               idx_closed=graph.idx_closed,
               flag_backpointers=False,
               flag_backpointers_cost=False)
    plt.plot(x_path[0, :], x_path[1, :], 'r', linewidth=2)
    print(graph.idx_closed)


def sphere_world_graph_test():
    '''
    Test of A* planner on discretized sphere world
    '''
    plt.figure()
    sphere_world = me570_graph.SphereWorldGraph(20)
    sphere_world.run_plot()


def twolink_search_test():
    '''
    Test of A* planner on the two-link manipulator
    '''
    my_robot_graph = me570_robot.TwoLinkGraph()
    my_robot_graph.plot()

    # Easy case
    # theta_start = np.vstack((0.76, 0.12))
    # theta_goal = np.vstack((0.76, 6.00))

    # Medium Case
    theta_start = np.vstack((0.76, 0.12))
    theta_goal = np.vstack((2.72, 5.45))

    theta_path = my_robot_graph.search_start_goal(theta_start, theta_goal)
    plt.plot(theta_path[0, :], theta_path[1, :], 'r')
    my_robot_graph.plot()

    plt.show()

    plt.figure()
    my_robot = me570_robot.TwoLink()
    my_robot.animate(theta_path)
