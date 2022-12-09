"""
Classes to define potential and potential planner for the sphere world
"""

import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import io as scio

import me570_geometry
import me570_qp as qp


class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different
        goal locations (one for each column).
        """
        data = scio.loadmat('sphereWorld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = sphere_args[1].item()
            sphere_args[2] = sphere_args[2].item()
            self.world.append(me570_geometry.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def plot(self):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a
        * marker at the goal location.
        """

        axes = plt.gca()
        for sphere in self.world:
            sphere.plot('r')

        axes.scatter(self.x_goal[0, :], self.x_goal[1, :], c='g', marker='*')

        axes.set_xlim([-11, 11])
        axes.set_ylim([-11, 11])
        plt.axis('equal')


class RepulsiveSphere:
    """ Repulsive potential for a sphere """
    def __init__(self, sphere):
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """s
        Evaluate the repulsive potential from  sphere at the location x= x_eval
        The function returns the repulsive potential as given by (eq:repulsive).
        """
        distance = self.sphere.distance(x_eval)

        distance_influence = self.sphere.distance_influence
        if distance > distance_influence:
            u_rep = 0
        elif distance_influence > distance > 0:
            u_rep = ((distance**-1 - distance_influence**-1)**2) / 2.
            u_rep = u_rep.item()
        else:
            u_rep = math.nan
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by (eq:repulsive-gradient).
        """
        distance = self.sphere.distance(x_eval)
        distance_influence = self.sphere.distance_influence
        if distance > distance_influence:
            grad_u_rep = np.zeros((2, 1))
        elif distance_influence > distance > 0:
            distance_grad = self.sphere.distance_grad(x_eval)
            grad_u_rep = -(distance**-1 - distance_influence**-1) * (
                distance**-2) * distance_grad
        else:
            grad_u_rep = np.zeros((2, 1))
            grad_u_rep[:] = np.nan
        return grad_u_rep


class Attractive:
    """ Repulsive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        if not isinstance(potential, dict) or not all(
                f in potential for f in ('x_goal', 'shape')):
            raise ValueError(
                '"potential" must be a dict with all expected keys')
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if shape == 'conic':
            expo = 1
        else:
            expo = 2
        u_attr = np.linalg.norm(x_eval - x_goal)**expo
        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval.
        The gradient is given by the formula If  potential['shape'] is equal to  'conic',
        use p=1; if it is equal to 'quadratic', use p=2.
        """
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if shape == 'conic':
            expo = 1
        else:
            expo = 2
        grad_u_attr = expo * (
            x_eval - x_goal) * np.linalg.norm(x_eval - x_goal)**(expo - 2)
        return grad_u_attr


class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """
        rep_weight = self.potential['repulsive_weight']
        attractive = Attractive(self.potential)
        attr_term = attractive.eval(x_eval)
        rep_term = 0
        worlds = self.world.world
        for sphere in worlds:
            repulsive = RepulsiveSphere(sphere)
            rep_term += repulsive.eval(x_eval)
        u_eval = attr_term + rep_weight * rep_term
        return u_eval

    def grad(self, x_eval):
        """
        Compute the gradient of the total potential,
        U=U_ attr+alpha sum_i U_ rep,i,, where alpha is given by
        the variable  potential.repulsiveWeight
        """
        rep_weight = self.potential['repulsive_weight']
        attractive = Attractive(self.potential)
        attr_grad_term = attractive.grad(x_eval)
        rep_grad_term = 0
        worlds = self.world.world
        for sphere in worlds:
            repulsive = RepulsiveSphere(sphere)
            rep_grad_term += repulsive.grad(x_eval)
        grad_u_eval = attr_grad_term + rep_weight * rep_grad_term
        return grad_u_eval


class Planner:
    """ A class implementing a generic potential planner and plot the results """
    def __init__(self, function, control, epsilon, nb_steps):
        """Save the arguments to internal attributes """
        self.function = function
        self.control = control
        self.epsilon = epsilon
        self.nb_steps = nb_steps

    def run(self, x_start):
        """
        This function uses a given function (planner_parameters['control']) to implement a generic
    potential-based planner with step size  planner_parameters['epsilon'], and evaluates the cost
    along the returned path. The planner must stop when either the number of steps given by
    planner_parameters['nb_steps'] is reached, or when the norm of the vector given by
    planner_parameters['control'] is less than 5 10^-3 (equivalently,  5e-3).
        """
        x_path = np.zeros((2, self.nb_steps))
        u_path = np.zeros((1, self.nb_steps))

        x_path[:] = np.nan
        u_path[:] = np.nan

        x_path[:, [0]] = x_start
        u_path[:, [0]] = self.function(x_path[:, [0]])

        for i_step in range(1, self.nb_steps):
            change = -self.control(x_path[:, [i_step - 1]])
            x_path[:,
                   [i_step]] = x_path[:, [i_step - 1]] + self.epsilon * change
            u_path[:, [i_step]] = self.function(x_path[:, [i_step]])
            norm = np.linalg.norm(self.control(x_path[:, [i_step]]), axis=0)
            if norm < 5e-3:
                break

        return x_path, u_path

    def run_plot(self):
        """
        This function performs the following steps:
     - Loads the problem data from the file !70!DarkSeaGreen2 sphereworld.mat.
     - For each goal location in  world.xGoal:
     - Uses the function Sphereworld.plot to plot the world in a first figure.
     - Calls the method run(). The function needs to be called five times, using
       each one of the initial locations
    given in  x_start (also provided in !70!DarkSeaGreen2 sphereworld.mat).
     - it:plot-plan After each call, plot the resulting trajectory superimposed to the world in the
    first subplot; in a second subplot, show  u_path (using the same color and using the  semilogy
    command).
        """
        sphere_world = SphereWorld()

        x_start = sphere_world.x_start
        nb_starts = x_start.shape[1]
        fig, axes = plt.subplots(ncols=3)
        for sub_axes in axes[0:2]:
            sub_axes.set_aspect('equal', adjustable='box')
            plt.sca(sub_axes)
            sphere_world.plot()
        axes[1].set_xlim((6.5, 8.5))
        axes[1].set_ylim((-1, 1))

        for start in range(0, nb_starts):
            x_start = sphere_world.x_start[:, [start]]
            x_path, u_path = self.run(x_start)
            for sub_axes in axes[0:2]:
                sub_axes.plot(x_path[0, :], x_path[1, :])
            axes[-1].plot(u_path.T)

        # Set auto y limits
        axes[-1].set_ylim(auto=True)

        # Make figure with good aspect ratio
        fig.set_size_inches([8, 2])


class Clfcbf_Control:
    """
    A class implementing a CLF-CBF-based control framework.
    """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes, and create an attribute
        attractive with an object of class  Attractive using the argument
        potential.
        """
        self.world = world
        self.potential = potential
        self.attractive = Attractive(potential)

    def function(self, x_eval):
        """
        Evaluate the CLF (i.e.,  self.attractive.eval()) at the given input.
        """
        return self.attractive.eval(x_eval)

    def control(self, x_eval):
        """
        Compute u^* according to (eq:clfcbf-qp).
        """
        c_cbf = self.potential['repulsive_weight']

        u_ref = self.attractive.grad(x_eval)

        nb_worlds = len(self.world.world)
        a_barrier = np.zeros((nb_worlds, 2))
        b_barrier = np.zeros((nb_worlds, 1))
        for i_obstacle, sphere in enumerate(self.world.world):
            a_barrier[i_obstacle, :] = sphere.distance_grad(x_eval).T
            b_barrier[i_obstacle, :] = sphere.distance(x_eval)
        b_barrier *= -c_cbf
        u_opt = qp.qp_supervisor(a_barrier, b_barrier, u_ref)

        return u_opt
