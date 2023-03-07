import cv2
import matplotlib.pyplot as plt
import numpy as np

dir = "./2.jpg"
src = cv2.imread(dir)
width = 4032
height = 3024

pts1 = np.float32([[460,310], [570,2340], [2590, 640], [2530,1820]])
pts2 = np.float32([[0,0], [0, height], [width, 0],[width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
homography = cv2.warpPerspective(src, matrix, (width, height))
#
rows = 2
cols = 2
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
for k in range(4):
    ax1.scatter(pts1[k][0], pts1[k][1])
ax1.set_title('Original Image')
ax1.axis("off")
ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(cv2.cvtColor(homography, cv2.COLOR_BGR2RGB))
ax2.set_title('Homography')
ax2.axis("off")

############################ Hough line Transform ########################################


gray = cv2.cvtColor(homography,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,100,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,150)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    houghline = cv2.line(homography,(x1,y1),(x2,y2),(0,0,255),1)
ax3 = fig.add_subplot(rows,cols,3)
ax3.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
ax3.set_title('Canny Edges')
ax3.axis("off")
ax4 = fig.add_subplot(rows,cols,4)
ax4.imshow(cv2.cvtColor(houghline, cv2.COLOR_BGR2RGB))
ax4.set_title('Hough Line Transform')
ax4.axis("off")
plt.show()
print(lines)



# pts1 = np.float32(coord_sort(points))
# pts2 = np.float32([[0, 0], [width, 0],[0,height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# img = img_to_coord(img)
# img = cv2.warpPerspective(img, matrix, (width, height))

