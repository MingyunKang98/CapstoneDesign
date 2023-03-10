import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

txt_dir = "./34.txt"

def txt_coord(dir):
    line_pd = pd.read_csv(dir , sep= " ", header=None) # 첫째행이 header가 아닌경우 header = None
    mask = line_pd[0]==2
    filtered_data =line_pd[mask]
    filtered_data = filtered_data.to_numpy()
    shp = filtered_data.shape
    filtered_data = filtered_data[:, 1:].reshape(shp[0],-1,2)
    filtered_data = filtered_data
    return filtered_data

coords = txt_coord(txt_dir)
width = 640
height = 640
plt.figure(figsize=(12,6))
for k in range(len(coords)):
    x = coords[k][:,0]
    x_mask = np.isnan(x)
    x = x[~x_mask] *width
    y = coords[k][:,1]
    y_mask = np.isnan(y)
    y= y[~y_mask] *height
    plt.plot(x, y)
    s = map(Point, zip(x,y))
    poly = Polygon(s)
    center = poly.centroid
    plt.text(center.x, center.y, round(poly.area, 2), fontsize="xx-small")
plt.show()