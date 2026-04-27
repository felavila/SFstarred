import numpy as np 
import matplotlib.pyplot as plt 

def find_nearest_object(array, coord):
        coord_rep = np.repeat(np.array(coord)[None, :], len(array), axis=0)
        metric = np.sqrt((array[:, 0]-coord_rep[:, 0])**2 + (array[:, 1]-coord_rep[:, 1])**2)
        idx = np.argmin(metric, axis=0)
        return array[idx]
    
def create_rectangle_patch(data,center, size=1.0):
    x, y = center
    width = height = size
    d = data[int(y - height / 2): int(y + height / 2),int(x - width / 2):int(x + width / 2)].copy()
    if np.all(np.isnan(d)):
          #print(np.nanmax(d))
          return plt.Rectangle((x - width / 2, y - height / 2), width, height, edgecolor='k', facecolor='none'), np.zeros((0,0)),[x, y]
    return plt.Rectangle((x - width / 2, y - height / 2), width, height, edgecolor='red', facecolor='none'), d,[x, y]#data[int(x_m - width / 2):int(x_m + width / 2), int(y_m - height / 2): int(y_m + height / 2)]


