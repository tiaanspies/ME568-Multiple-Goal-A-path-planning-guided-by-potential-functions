"""
 Please merge the functions and classes from this file with the same file from
the previous homework assignment
"""
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scio

import me570_geometry as gm
import me570_potential as pot
from me570_graph import grid2graph
from me570_potential import SphereWorld


def hat2(theta):
    """
    Given a scalar, return the 2x2 skew-symmetric matrix corresponding to the
    hat operator
    """
    return np.array([[0, -theta], [theta, 0]])


def polygons_add_x_reflection(vertices):
    """
    Given a sequence of vertices, adds other vertices by reflection
    along the x axis
    """
    vertices = np.hstack([vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
    return vertices


def polygons_generate():
    """
    Generate the polygons to be used for the two-link manipulator
    """
    vertices1 = np.array([[0, 5], [-1.11, -0.511]])
    vertices1 = polygons_add_x_reflection(vertices1)
    vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                          [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
    vertices2 = polygons_add_x_reflection(vertices2)
    return (gm.Polygon(vertices1), gm.Polygon(vertices2))


polygons = polygons_generate()


class TwoLink:
    '''Class for Two Link Manipulator'''
    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the
        vertices of the links, all transformed according to  _1, _2.
        """

        # Rotation matrices
        rotation_w_b1 = gm.rot2d(theta[0, 0])
        rotation_b1_b2 = gm.rot2d(theta[1, 0])
        rotation_w_b2 = rotation_w_b1 @ rotation_b1_b2

        # Translation matrix
        translation_b1_b2 = np.vstack((5, 0))
        translation_w_b2 = rotation_w_b1 @ translation_b1_b2

        # Transform end effector from B₂ to W
        p_eff_b2 = np.vstack((5, 0))
        vertex_effector_transf = rotation_w_b2 @ p_eff_b2 + translation_w_b2

        # Transform polygon 1 from B₁ to W
        polygon1_vertices_b1 = polygons[0].vertices
        polygon1_transf = gm.Polygon(rotation_w_b1 @ polygon1_vertices_b1)

        # Transform polygon 2 from B₂ to W
        polygon2_vertices_b2 = polygons[1].vertices
        polygon2_transf = gm.Polygon(rotation_w_b2 @ polygon2_vertices_b2 +
                                     translation_w_b2)
        return vertex_effector_transf, polygon1_transf, polygon2_transf

    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        [_, polygon1_transf, polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(color)
        polygon2_transf.plot(color)

    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """

        nb_theta = theta.shape[1]
        flag_theta = [False] * nb_theta

        for i_theta in range(theta.shape[1]):
            config = np.vstack(theta[:, i_theta])
            [_, polygon1_transf, polygon2_transf] = self.kinematic_map(config)

            flag_points1 = polygon1_transf.is_collision(points)
            flag_points2 = polygon2_transf.is_collision(points)
            flag_theta[i_theta] = any(flag_points1) or any(flag_points2)

        return flag_theta

    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.
        """
        collisions = self.is_collision(theta, points)
        for i, is_collision in enumerate(collisions):
            if is_collision:
                color = 'r'
            else:
                color = 'g'

            self.plot(np.vstack(theta[:, i]), color)
            plt.scatter(points[0, :], points[1, :], c='k', marker='*')

    def animate(self, theta):
        """
        Draw the two-link manipulator for each column in theta
        """
        theta_steps = theta.shape[1]
        for i_theta in range(0, theta_steps, 15):
            self.plot(theta[:, [i_theta]], 'k')

    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """

        vertex_effector_dot = [[], []]
        t_columnsize = theta.shape[1]
        for i in range(t_columnsize):
            theta1 = theta[0, i]
            theta2 = theta[1, i]
            jacobian = np.array(
                [[
                    -5 * math.sin(theta1 + theta2) - 5 * math.sin(theta1),
                    -5 * math.sin(theta1 + theta2)
                ],
                 [
                     5 * math.cos(theta1 + theta2) + 5 * math.cos(theta1),
                     5 * math.cos(theta1 + theta2)
                 ]])
            effector_eval = jacobian @ theta_dot

            vertex_effector_dot = np.concatenate(
                (vertex_effector_dot, effector_eval), axis=1)
        return vertex_effector_dot

    def jacobian_matrix(self, theta):
        """
        Compute the matrix representation of the Jacobian of the position of the end effector with
    respect to the joint angles as derived in Question~ q:jacobian-matrix.
        """
        theta1 = theta[0, :].item()
        theta2 = theta[1, :].item()
        jtheta = np.array(
            [[
                -5 * math.sin(theta1 + theta2) - 5 * math.sin(theta1),
                -5 * math.sin(theta1 + theta2)
            ],
             [
                 5 * math.cos(theta1 + theta2) + 5 * math.cos(theta1),
                 5 * math.cos(theta1 + theta2)
             ]])
        return jtheta


class TwoLinkPotential:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential
        self.world = world

    def eval(self, theta_eval):
        """
        Compute the potential U pulled back through the kinematic map of the
        two-link manipulator, i.e.,  U(  Wp_ eff(  )), where U is defined as in
        Question~ q:total-potential, and Wp_ eff( ) is the position of the end
        effector in the world frame as a function of the joint
        angles theta1,theta2
        """
        sphere_world = self.world
        potential = self.potential
        theta1 = theta_eval[0, :].item()
        theta2 = theta_eval[1, :].item()
        rot_theta1 = gm.rot2d(theta1)
        rot_theta2 = gm.rot2d(theta2)
        endeff_b2 = np.array([[5], [0]])
        translation_b12 = np.array([[5], [0]])
        w_peff = rot_theta1 @ rot_theta2 @ endeff_b2 + rot_theta1 @ translation_b12
        total = pot.Total(sphere_world, potential)
        u_eval_theta = total.eval(w_peff)
        return u_eval_theta

    def grad(self, theta_eval):
        """
        Compute the gradient of the potential U pulled back through the kinematic
        map of the two-link manipulator, i.e.,  _   U(  Wp_ eff(  )).
        """
        sphere_world = self.world
        potential = self.potential
        theta1 = theta_eval[0, :]
        theta2 = theta_eval[1, :]
        rot_theta1 = gm.rot2d(theta1)
        rot_theta2 = gm.rot2d(theta2)
        endeff_b2 = np.array([[5], [0]])
        translation_b12 = np.array([[5], [0]])
        w_peff = rot_theta1 @ rot_theta2 @ endeff_b2 + rot_theta1 @ translation_b12
        total = pot.Total(sphere_world, potential)
        two_links = TwoLink()
        jacobian = two_links.jacobian_matrix(theta_eval)

        grad_u_eval_theta = jacobian.T @ total.grad(w_peff)
        return grad_u_eval_theta

    def run_plot(self, epsilon, nb_steps):
        """
        This function performs the same steps as Planner.run_plot in a previous question
    except for the following:
     - In step  it:grad-handle:  planner_parameters['U'] should be set to  @twolink_total, and
    planner_parameters['control'] to the negative of  @twolink_totalGrad.
     - In step  it:grad-handle: Use the contents of the variable  thetaStart instead of  xStart to

        initialize the planner, and use only the second goal  x_goal[:,1].
     - In step  it:plot-plan: Use Twolink.plotAnimate to plot a decimated version of the results of
    the planner. Note that the output  xPath from Potential.planner will really contain a sequence
    of join angles, rather than a sequence of 2-D points. Plot only every 5th or 10th column of
    xPath (e.g., use  xPath(:,1:5:end)). To avoid clutter, plot a different figure for each start.
        """

        sphere_world = SphereWorld()

        nb_starts = sphere_world.theta_start.shape[1]

        planner = pot.Planner(function=self.eval,
                              control=self.grad,
                              epsilon=epsilon,
                              nb_steps=nb_steps)

        two_link = TwoLink()

        for start in range(0, nb_starts):
            # Run the planner
            theta_start = sphere_world.theta_start[:, [start]]
            theta_path, u_path = planner.run(theta_start)

            # Plots
            _, axes = plt.subplots(ncols=2)
            axes[0].set_aspect('equal', adjustable='box')
            plt.sca(axes[0])
            sphere_world.plot()
            two_link.animate(theta_path)
            axes[1].plot(u_path.T)


class TwoLinkGraph:
    """
    A class for finding a path for the two-link manipulator among given obstacle points using a grid
discretization and  A^*.
    """
    def __init__(self) -> None:
        self.graph = self.load_free_space_graph()

    def load_free_space_graph(self):
        """
        The function performs the following steps
         - Calls the method load_free_space_grid.
         - Calls grid2graph.
         - Stores the resulting  graph object of class  Grid as an internal attribute.
        """
        grid = load_free_space_grid()
        return grid2graph(grid)

    def plot(self):
        """
        Use the method Graph.plot to visualize the contents of the attribute  graph.
        """
        self.graph.plot(flag_backpointers_cost=False)

    def search_start_goal(self, theta_start, theta_goal):
        """
        Use the method Graph.search to search a path in the graph stored in  graph.
        """
        theta_path = self.graph.search_start_goal(theta_start, theta_goal)
        return theta_path


def load_free_space_grid():
    """
Loads the contents of the file ! twolink_freeSpace_data.mat
    """
    test_data = scio.loadmat('twolink_freeSpace_data.mat')
    test_data = test_data['grid'][0][0]
    grid = gm.Grid(test_data[0][0], test_data[1][0])

    grid.fun_evalued = test_data[2]
    return grid
