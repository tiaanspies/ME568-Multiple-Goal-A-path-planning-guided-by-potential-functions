"""
Classes and utility functions for working with graphs (plotting, search, initialization, etc.)
"""
from math import pi

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import me570_geometry as geo
import me570_queue as queue


def plot_arrows_from_list(arrow_list, scale=1.0, color=(0., 0., 0.)):
    """
    Plot arrows from a list of pairs of base points and displacements
    """
    x_edges, v_edges = [np.hstack(x) for x in zip(*arrow_list)]
    plt.quiver(x_edges[0, :],
               x_edges[1, :],
               v_edges[0, :],
               v_edges[1, :],
               angles='xy',
               scale_units='xy',
               scale=scale,
               color=color)


def plot_text(coord, str_label, color=(1., 1., 1.)):
    """
    Wrap plt.text to get a consistent look
    """
    plt.text(coord[0].item(),
             coord[1].item(),
             str_label,
             ha="center",
             va="center",
             fontsize='xx-small',
             bbox=dict(boxstyle="round", fc=color, ec=None))


class Graph:
    """
    A class collecting a  graph_vector data structure and all the functions that operate on a graph.
    """
    def __init__(self, graph_vector):
        """
        Stores the arguments as internal attributes.
        neighbors
        neighbors_cost
        g
        backpointer
        x
        """
        self.graph_vector = graph_vector
        # for vec in self.graph_vector:
        #     vec['g'] = 0.0
        #     vec['backpointer'] = None

    def _apply_neighbor_function(self, func):
        """
        Apply a function on each node and chain the result
        """
        list_of_lists = [func(n) for n in self.graph_vector]
        return [e for l in list_of_lists for e in l]

    def _neighbor_weights_with_positions(self, n_current):
        """
        Get all weights and where to display them
        """
        x_current = n_current['x']
        return [
            (weight_neighbor,
             self.graph_vector[idx_neighbor]['x'] * 0.25 + x_current * 0.75)
            for (weight_neighbor, idx_neighbor
                 ) in zip(n_current['neighbors_cost'], n_current['neighbors'])
        ]

    def _neighbor_displacements(self, n_current):
        """
        Get all displacements with respect to the neighbors for a given node
        """
        x_current = n_current['x']
        return [(x_current, self.graph_vector[idx_neighbor]['x'] - x_current)
                for idx_neighbor in n_current['neighbors']]

    def _neighbor_backpointers(self, n_current):
        """
        Get coordinates for backpointer arrows
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [
                (x_current,
                 0.5 * (self.graph_vector[idx_backpointer]['x'] - x_current))
            ]
        else:
            arrow = []
        return arrow

    def _neighbor_backpointers_cost(self, n_current):
        """
        Get value and coordinates for backpointer costs
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [(n_current['g'],
                      self.graph_vector[idx_backpointer]['x'] * 0.25 +
                      x_current * 0.75)]
        else:
            arrow = []
        return arrow

    def has_backpointers(self):
        """
        Return True if self.graph_vector has a "backpointer" field
        """
        return self.graph_vector is not None and len(
            self.graph_vector) > 0 and 'backpointer' in self.graph_vector[0]

    def plot(self,
             flag_edges=True,
             flag_labels=False,
             flag_edge_weights=False,
             flag_backpointers=True,
             flag_backpointers_cost=True,
             flag_heuristic=False,
             node_lists=None,
             idx_goal=None):
        """
        The function plots the contents of the graph described by the  graph_vector structure,
        alongside other related, optional data.
        """

        if flag_edges:
            displacement_list = self._apply_neighbor_function(
                self._neighbor_displacements)
            plot_arrows_from_list(displacement_list, scale=1.05)

        if flag_labels:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                plot_text(x_current, str(idx))

        if idx_goal is not None:
            x_goal = self.graph_vector[idx_goal]['x']
            plt.plot(x_goal[0, :],
                     x_goal[1, :],
                     marker='d',
                     markersize=10,
                     color=(.8, .1, .1))

        if flag_heuristic and idx_goal is not None:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                h_current = self.heuristic(idx, idx_goal)
                plot_text(x_current, 'h=' + str(h_current), color=(.8, 1., .8))

        if flag_edge_weights:
            weight_list = self._apply_neighbor_function(
                self._neighbor_weights_with_positions)
            for (weight, coord) in weight_list:
                plot_text(coord, str(weight), color=(.8, .8, 1.))

        if flag_backpointers and self.has_backpointers():
            backpointer_arrow_list = self._apply_neighbor_function(
                self._neighbor_backpointers)
            plot_arrows_from_list(backpointer_arrow_list,
                                  scale=1.05,
                                  color=(0.1, .8, 0.1))

        if flag_backpointers_cost and self.has_backpointers:
            backpointer_cost_list = self._apply_neighbor_function(
                self._neighbor_backpointers_cost)
            for (cost, coord) in backpointer_cost_list:
                plot_text(coord, 'g=' + str(cost), color=(.8, 1., .8))

        if node_lists is not None:
            if not isinstance(node_lists[0], list):
                node_lists = [node_lists]
            markers = ['d', 'o', 's', '*', 'h', '^', '8']
            for i, lst in enumerate(node_lists):
                x_list = [self.graph_vector[e]['x'] for e in lst]
                coords = np.hstack(x_list)
                plt.plot(
                    coords[0, :],
                    coords[1, :],
                    markers[i % len(markers)],
                    markersize=10,
                )

    def nearest_neighbors(self, x_query, k_nearest):
        """
        Returns the k nearest neighbors in the graph for a given point.
        """

        x_graph = np.hstack([n['x'] for n in self.graph_vector])
        distances_squared = np.sum((x_graph - x_query)**2, 0)
        idx = np.argpartition(distances_squared, k_nearest)
        return idx[:k_nearest]

    def attractive_pot(self, x_pt, x_entrance):
        # Linear attractive potential
        # inputs id of x_pt and coordinated od x_entrance
        x = self.graph_vector[x_pt]['x']
        return np.linalg.norm(x - x_entrance, 2)

    def attractive_pot_cartesian(self, x, x_entrance):
        # Linear attractove potential 
        return np.linalg.norm(x - x_entrance, 2)

    def repulsive_pot(self, x_pt, obstacles):
        # repulsive potential based on nearest obstacle
        DIST_INF = 6

        min_dist = np.inf
        x = self.graph_vector[x_pt]['x']
        for obstacle in obstacles:
            dist = self.lineseg_dists(x, obstacle[0], obstacle[1])

            if np.min(dist) < min_dist:
                min_dist = np.min(dist)

        if min_dist > DIST_INF:
            u_rep = 0
        elif DIST_INF > min_dist > 0:
            u_rep = ((min_dist**-1 - DIST_INF**-1)**2)
            # u_rep = 1/min_dist**2
        else:
            u_rep = 0

        return u_rep

    def repulsive_pot_cartesian(self, x, obstacles):
        DIST_INF = 6

        min_dist = np.inf
        for obstacle in obstacles:
            dist = self.lineseg_dists(x, obstacle[0], obstacle[1])

            if np.min(dist) < min_dist:
                min_dist = np.min(dist)

        if min_dist > DIST_INF:
            u_rep = 0
        elif DIST_INF > min_dist > 0:
            # u_rep = 5/min_dist**2
            u_rep = ((min_dist**-1 - DIST_INF**-1)**2)
        else:
            u_rep = 0

        return u_rep

    def lineseg_dists(self, p, a, b):
        """Cartesian distance from point to line segment
        ADAPTED FROM 
        "https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line"
        """
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[0, :], d_ba[1, :]).reshape(1, -1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=0)
        t = np.multiply(p - b, d).sum(axis=0)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors  
        d_pa = p - a
        c = d_pa[0, :] * d[1, :] - d_pa[1, :] * d[0, :]

        return np.hypot(h, c)

    def plot_attractive(self, x_entrance):
        xx = np.reshape(np.linspace(0, 50, 20), (20, 1))
        xx = np.repeat(xx, 20, axis=1)
        yy = xx.copy().T
        zz = np.zeros(shape=(20, 20))
        for x in range(20):
            for y in range(20):
                pos = np.array([[xx[x, 0]],[yy[0, y]]])
                a = self.attractive_pot_cartesian(pos, x_entrance)
                zz[x, y] = a

        # Creating plot
        fig, ax = plt.subplots()
        cmap = mpl.cm.viridis
        c = ax.pcolormesh(xx, yy, zz, cmap=cmap)
        fig.colorbar(c, ax=ax)

        # show plot
        plt.show()

    def plot_repulsive(self, obstacles):
        xx = np.reshape(np.linspace(0, 50, 101), (101, 1))
        xx = np.repeat(xx, 101, axis=1)
        yy = xx.copy().T
        zz = np.zeros(shape=(101, 101))
        for x in range(101):
            for y in range(101):
                pos = np.array([[xx[x, 0]],[yy[0, y]]])
                a = self.repulsive_pot_cartesian(pos, obstacles)
                zz[x, y] = a
        # Creating plot
        plt.imshow(np.rot90(zz, axes=(0, 1)), cmap='viridis')
        plt.colorbar()
        plt.show()

    def plot_total(self, obstacles, x_entrance):
        xx = np.reshape(np.linspace(0, 50, 101), (101, 1))
        xx = np.repeat(xx, 101, axis=1)
        yy = xx.copy().T
        zz = np.zeros(shape=(101, 101))
        for x in range(101):
            for y in range(101):
                pos = np.array([[xx[x, 0]],[yy[0, y]]])
                a = self.repulsive_pot_cartesian(pos, obstacles)
                b = self.attractive_pot_cartesian(pos, x_entrance)
                zz[x, y] = 30*a+3*b

        plt.imshow(np.rot90(zz, axes=(0, 1)), cmap='viridis')
        plt.colorbar()
        plt.show()
    
    def heuristic_total(self, idx_x, idx_goals, x_entrance, obstacles):
        # joiing the normal heuristic with the attractive and repulsive potentials
        h = self.heuristic(idx_x, idx_goals)
        f = self.attractive_pot(idx_x, x_entrance)
        j = self.repulsive_pot(idx_x, obstacles)

        return h + 3*f+ 30*j

    def heuristic(self, idx_x, idx_goals):
        """
        Computes the heuristic  h given by the Euclidean distance between the nodes with indexes
        idx_x and idx_goal.
        """
        new_arr = np.zeros(shape=(2, idx_goals.shape[0]))
        for i, goal in enumerate(idx_goals):
            new_arr[:, i, None] = self.graph_vector[goal]['x']

        diff = new_arr - self.graph_vector[idx_x]['x']
        h_val = np.linalg.norm(diff, 2, axis=0)
        return np.min(h_val)

    def get_expand_list(self, idx_n_best, idx_closed):
        """
        Finds the neighbors of element  idx_n_best that are not in  idx_closed (line   in Algorithm~
        ).
        """
        idx_expand = []

        for neighbor in self.graph_vector[idx_n_best]['neighbors']:
            if idx_closed is None:
                idx_expand.append(neighbor)
            elif neighbor not in idx_closed:
                idx_expand.append(neighbor)
        return idx_expand

    def expand_element(self, idx_n_best, idx_x, idx_goals, pq_open, x_entrance, obstacles):
        """
        This function expands the vertex with index  idx_x (which is a neighbor of the one with
        index  idx_n_best) and returns the updated versions of  graph_vector and  pq_open.
        """

        vec_best = self.graph_vector[idx_n_best]
        cost = vec_best['neighbors_cost'][vec_best['neighbors'].index(idx_x)]

        g_best = self.graph_vector[idx_n_best]['g']
        

        if not pq_open.is_member(idx_x):
            self.graph_vector[idx_x]['g'] = g_best + cost
            self.graph_vector[idx_x]['backpointer'] = idx_n_best
            heur = self.heuristic_total(idx_x, idx_goals, x_entrance, obstacles)
            # heur = self.heuristic(idx_x, idx_goals)
            pq_open.insert(idx_x, g_best + cost + heur)

        elif g_best + cost < self.graph_vector[idx_x]['g']:
            self.graph_vector[idx_x]['g'] = g_best + cost
            self.graph_vector[idx_x]['backpointer'] = idx_n_best

        return pq_open

    def path(self, idx_start, idx_goal):
        """
        This function follows the backpointers from the node with index  idx_goal in  graph_vector
        to the one with index  idx_start node, and returns the  coordinates (not indexes) of the
        sequence of traversed elements.
        """
        idx_current = idx_goal
        x_path = np.array(self.graph_vector[idx_goal]['x'])

        while idx_current != idx_start:
            idx_current = self.graph_vector[idx_current]['backpointer']
            x_path = np.hstack([x_path, self.graph_vector[idx_current]['x']])

        return x_path

    def search(self, idx_start, idx_goal, idx_goals, x_entrance, obstacles):
        """
        Implements the  A^* algorithm, as described by the pseudo-code in Algorithm~ .
        """
        pq_open = queue.PriorityQueue()
        pq_open.insert(idx_start, 0)

        # self.graph_vector[idx_start]['backpointer'] = None
        # self.graph_vector[idx_start]['g'] = 0.0
        for vec in self.graph_vector:
            vec['g'] = 0.0
            vec['backpointer'] = None
        pq_closed = []

        index = 0

        while len(pq_open.queue_list) > 0:
            index += 1
            idx_n_best, _ = pq_open.min_extract()

            # Plot the points on the graph as they are searched
            plt.plot(self.graph_vector[idx_n_best]["x"][0], self.graph_vector[idx_n_best]["x"][1], 'r.')
            plt.draw()
            figure = plt.gcf()
            figure.canvas.flush_events()

            pq_closed.append(idx_n_best)

            if idx_n_best in idx_goals:
                idx_goal_found = idx_n_best
                break

            for x in self.get_expand_list(idx_n_best, pq_closed):
                self.expand_element(idx_n_best, x, idx_goals, pq_open, x_entrance, obstacles)

        x_path = self.path(idx_start, idx_goal_found)

        return x_path

    def search_start_goal(self, x_start, x_goals, x_entrance, obstacles):
        """
        This function performs the following operations:
         - Identifies the two indexes  idx_start,  idx_goal in  graph.graph_vector that are closest
        to  x_start and  x_goal (using Graph.nearestNeighbors twice, see Question~ -nearest).
         - Calls Graph.search to find a feasible sequence of points  x_path from  idx_start to
        idx_goal.
         - Appends  x_start and  x_goal, respectively, to the beginning and the end of the array
        x_path.
        """
        idx_start = self.nearest_neighbors(x_start, 1)[0]
        idx_goals = np.zeros(shape=(0))
        for goal in x_goals.T:
            x_goal = np.reshape(goal, (2,1))
            idx_goals = np.append(idx_goals, self.nearest_neighbors(x_goal, 1)[0])
        
        idx_goals = idx_goals.astype(int)

        # add parking spots to neighbors of final goal.
        final_goal_idx = np.shape(self.graph_vector)[0]
        x = np.array([[-10e10], [-10e10]])

        costs = np.zeros(shape=idx_goals.shape)
        vector_last = {"x": x, "neighbors": idx_goals, "neighbors_cost":costs}
        self.graph_vector = np.append(self.graph_vector, vector_last)


        for idx_goal in idx_goals:
            self.graph_vector[idx_goal]["neighbors"].append(final_goal_idx)
            self.graph_vector[idx_goal]["neighbors_cost"].append(0)

        x_path = self.search(idx_start, final_goal_idx, idx_goals, x_entrance, obstacles)

        # x_path = np.hstack([x_path, x_start])
        # x_path = np.hstack([x_goal, x_path])

        return x_path


def grid2graph(grid):
    """
    The function returns a  Graph object described by the inputs. See Figure~  for an example of the
    expected inputs and outputs.
    """

    # Make sure values in F are logicals
    fun_evalued = np.vectorize(bool)(grid.fun_evalued)

    # Get number of columns, rows, and nodes
    nb_xx, nb_yy = fun_evalued.shape
    nb_nodes = np.sum(fun_evalued)

    # Get indeces of non-zero entries, and assign a progressive number to each
    idx_xx, idx_yy = np.where(fun_evalued)
    idx_assignment = range(0, nb_nodes)

    # Lookup table from xx,yy element to assigned index (-1 means not assigned)
    idx_lookup = -1 * np.ones(fun_evalued.shape, 'int')
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        idx_lookup[i_xx, i_yy] = i_assigned

    def grid2graph_neighbors(idx_xx, idx_yy):
        """
        Finds the neighbors of a given element
        """

        displacements = [(idx_xx + dx, idx_yy + dy) for dx in [-1, 0, 1]
                         for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        neighbors = []
        for i_xx, i_yy in displacements:
            if 0 <= i_xx < nb_xx and 0 <= i_yy < nb_yy and idx_lookup[
                    i_xx, i_yy] >= 0:
                neighbors.append(idx_lookup[i_xx, i_yy].item())

        return neighbors

    # Create graph_vector data structure and populate 'x' and 'neighbors' fields
    graph_vector = [None] * nb_nodes
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        x_current = np.array([[grid.xx_grid[i_xx]], [grid.yy_grid[i_yy]]])
        neighbors = grid2graph_neighbors(i_xx, i_yy)
        graph_vector[i_assigned] = {'x': x_current, 'neighbors': neighbors}

    # Populate the 'neighbors_cost' field
    # Cannot be done in the loop above because not all 'x' fields would be filled
    for idx, n_current in enumerate(graph_vector):
        x_current = n_current['x']

        if len(n_current['neighbors']) > 0:
            x_neighbors = np.hstack(
                [graph_vector[idx]['x'] for idx in n_current['neighbors']])
            neighbors_cost_np = np.sum((x_neighbors - x_current)**2, 0)
            graph_vector[idx]['neighbors_cost'] = list(neighbors_cost_np)
        else:
            graph_vector[idx]['neighbors_cost'] = []

    return Graph(graph_vector)


# def test_nearest_neighbors():
#     """
#     Tests Graph.nearest_neighbors by picking a random point and finding the 3 nearest neighbors
#     in graphVectorMedium
#     """
#     graph = Graph(graph_load_test_data('graphVectorMedium'))
#     x_query = np.array([[5], [4]]) * np.random.rand(2, 1)
#     idx_neighbors = graph.nearest_neighbors(x_query, 3)
#     graph.plot(node_lists=idx_neighbors)
#     plt.scatter(x_query[[0]], x_query[[1]])


def test_grid2graph():
    """
    Tests grid2graph() by creating an arbitrary function returning bools
    """
    xx_grid = np.linspace(0, 2 * pi, 40)
    yy_grid = np.linspace(0, pi, 20)
    def func(x):
        return (x[[1]] > pi / 2 or np.linalg.norm(x - np.ones(
        (2, 1))) < 0.75) and np.linalg.norm(x - np.array([[4], [2.5]])) >= 0.5


    # func = lambda x: (x[[1]] > pi / 2 or np.linalg.norm(x - np.ones(
    #     (2, 1))) < 0.75) and not np.linalg.norm(x - np.array([[4], [2.5]])
    #                                             ) < 0.5
    grid = geo.Grid(xx_grid, yy_grid)
    grid.eval(func)
    graph = grid2graph(grid)
    graph.plot()
