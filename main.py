import me570_geometry as geo
import me570_graph as Graph
import numpy as np
from matplotlib import pyplot as plt
import carParking as cp

def getGrid(img):
    res = (img[:, :] == np.ones((1,3))*255).all(axis=2)
    return res


def main():
    car = cp.Car(length=10, width=6, steeringAngle=(np.pi)/3)
    carTranslated = car.transform(np.array([np.pi/2, 1]), np.array([[5, 5], [75, 20]]))

    xx = np.linspace(0, 50, 51)
    yy = np.linspace(0, 50, 51)
    grid = geo.Grid(xx, yy)

    grid_colors = np.ones((51, 51, 3), dtype=np.uint8)*255
    black = np.array((0, 0, 0))

    for i in range(51):
        grid_colors[0, i] = black
        grid_colors[50, i] = black
        grid_colors[i, 0] = black
        grid_colors[i, 50] = black
    
    
    for j in range(8):
        grid_colors[10:16, j*5] = black
        grid_colors[25:37, j*5] = black

    grid_colors[10, 5:35] = black
    grid_colors[31, 5:35] = black

    grid.fun_evalued = np.transpose(getGrid(grid_colors)) 

    X_start = np.array([[5], [5]])
    parking_spots = np.array([[45, 30, 20], [45, 45, 45]])

    graph = Graph.grid2graph(grid)
    path = graph.search_start_goal(X_start, parking_spots)


    plt.imshow(grid_colors)    
    plt.plot(parking_spots[0, :], parking_spots[1, :], "g*")
    # graph.plot()
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    plt.plot(path[0, :], path[1, :])
    plt.show()
    

if __name__ == "__main__":
    main()