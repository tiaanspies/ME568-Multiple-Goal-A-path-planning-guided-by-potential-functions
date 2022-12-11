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
    for j in range(8):
        grid_colors[10:16, j*5+5] = black
        grid_colors[25:37, j*5+5] = black
        park_spot = np.array([[7+j*5], [13]])

        if j < 7:
            parking_spots = np.hstack([parking_spots, park_spot])

    grid_colors[10, 5:40] = black
    grid_colors[31, 5:40] = black

    # grid_colors[10, 0:5] = black

    # create polygon obstacle around border of parking spots
    obstacle_1 = np.array([[5, 5, 40, 40], [10, 16, 16, 10]])
    obstacle_2 = np.array([[5, 5, 40, 40], [25, 37, 37, 25]])
    obstacles = np.array([obstacle_1, obstacle_2])

    #Rotate grid to match graph type
    grid.fun_evalued = np.transpose(getGrid(grid_colors))

    graph = Graph.grid2graph(grid) 

    # Plot map
    plt.imshow(grid_colors) 

    return graph, obstacles, parking_spots


def main():
    # CAR model
    # car = cp.Car(length=10, width=6, steeringAngle=(np.pi)/3)
    # carTranslated = car.transform(np.array([np.pi/2, 1]), np.array([[5, 5], [75, 20]]))

    X_start = np.array([[30], [10]])
    entrance = np.array([[0], [25]])

    graph, obstacles, parking_spots = create_map()

    # graph.plot()
    # plt.show()

    path = graph.search_start_goal(X_start, parking_spots, entrance, obstacles)

    plt.plot(parking_spots[0, :], parking_spots[1, :], "g*")
    plt.plot(entrance[0], entrance[1], 'or')
    # graph.plot()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    plt.plot(path[0, :], path[1, :])
    plt.show()

    # graph.plot_attractive(entrance)
    graph.plot_repulsive(obstacles)
   
if __name__ == "__main__":
    main()