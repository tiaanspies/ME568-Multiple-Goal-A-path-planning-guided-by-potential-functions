import me570_geometry as geo
import me570_graph as Graph
import numpy as np
from matplotlib import pyplot as plt
import carParking as cp

def getGrid(img):
    res = (img[:, :] == np.ones((1,3))*255).all(axis=2)
    return res

def create_map():
    # Create GRID
    xx = np.linspace(0, 50, 51)
    yy = np.linspace(0, 50, 51)
    grid = geo.Grid(xx, yy)

    grid_colors = np.ones((51, 51, 3), dtype=np.uint8)*255
    black = np.array((0, 0, 0))

    # Create outer border
    for i in range(51):
        grid_colors[0, i] = black
        grid_colors[50, i] = black
        grid_colors[i, 0] = black
        grid_colors[i, 50] = black
    
    # Create Parking Spots
    parking_spots = np.zeros(shape=(2,0))
    spot_count = 7
    for j in range(spot_count):
        grid_colors[10:16, j*6+5] = black
        grid_colors[25:37, j*6+5] = black
        park_spot1 = np.array([[8+j*6], [11]])
        park_spot2 = np.array([[8+j*6], [30]])
        park_spot3 = np.array([[8+j*6], [32]])

        if j < spot_count-1:
            parking_spots = np.hstack([parking_spots, park_spot1, park_spot2, park_spot3])

    grid_colors[10, 5:42] = black
    grid_colors[31, 5:42] = black

    # BLOCK OFF TWO OF THE PARKING SPOTS
    # grid_colors[25, 5:23] = black
    # obstacle = np.zeros(shape=(2,2,1))
    # obstacle[0,:,None,0] = np.array([[5], [25]])
    # obstacle[1,:,None,0] = np.array([[23], [25]])

    # obstacle_3 = obstacle.copy()

    # Create line obstacles that match the parking spots for repulsive potential
    obstacle = np.zeros(shape=(2,2,spot_count+1))
    for j in range(spot_count):
        obstacle[0,:, None,j] = np.array([[j*6+5], [10]])
        obstacle[1,:, None,j] = np.array([[j*6+5], [16]])

    obstacle[0,:,None,spot_count] = np.array([[5], [10]])
    obstacle[1,:,None,spot_count] = np.array([[41], [10]])

    obstacle_1 = obstacle.copy()

    obstacle = np.zeros(shape=(2,2,spot_count+1))
    for j in range(spot_count):
        obstacle[0,:, None,j] = np.array([[j*6+5], [25]])
        obstacle[1,:, None,j] = np.array([[j*6+5], [37]])

    obstacle[0,:,None,spot_count] = np.array([[5], [31]])
    obstacle[1,:,None,spot_count] = np.array([[41], [31]])

    obstacle_2 = obstacle.copy()
    # # create polygon obstacle around border of parking spots
    # obstacle_1 = np.array([[5, 5, 42, 42], [10, 16, 16, 10]])
    # obstacle_2 = np.array([[5, 5, 42, 42], [25, 37, 37, 25]])
    obstacles = [obstacle_1, obstacle_2]

    #Rotate grid to match graph type
    grid.fun_evalued = np.transpose(getGrid(grid_colors))

    graph = Graph.grid2graph(grid) 

    return graph, obstacles, parking_spots, grid_colors


def main():
    X_start = np.array([[30], [49]])
    # X_start = np.array([[45], [15]])
    # X_start = np.array([[45], [2]])
    # X_start = np.array([[45], [35]])
    # X_start = np.array([[45], [32]])
    entrance = np.array([[0], [25]])

    graph, obstacles, parking_spots, grid_colors = create_map()

    # Plot map
    
    plt.imshow(grid_colors) 
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    plt.plot(parking_spots[0, :], parking_spots[1, :], "g*")
    plt.plot(entrance[0], entrance[1], color='orange',marker='.',markersize=40)
    plt.show()

    # Plot the node expanding
    plt.imshow(grid_colors) 
    plt.ion()
    # graph.plot()
    plt.plot(parking_spots[0, :], parking_spots[1, :], "g*")
    plt.plot(entrance[0], entrance[1], color='orange',marker='.',markersize=40)

    # graph.plot()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    path = graph.search_start_goal(X_start, parking_spots, entrance, obstacles)
    plt.ioff()
    
    plt.show()
    plt.imshow(grid_colors)
    plt.plot(path[0, :], path[1, :])
    plt.plot(parking_spots[0, :], parking_spots[1, :], "g*")
    plt.plot(entrance[0], entrance[1], color='orange',marker='.',markersize=40)

    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    # graph.plot_attractive(entrance)
    # graph.plot_attractive(entrance)
    # graph.plot_repulsive(obstacles)
    # graph.plot_total(obstacles, entrance)
    plt.show()
   
if __name__ == "__main__":
    main()