import cv2
import matplotlib.pyplot as plt
import numpy as np

dir = "./2.jpg"
src = cv2.imread(dir)

print(np.shape(src))
width = np.shape(src)[0]
height = np.shape(src)[1]
pts1 = np.float32([[460,310], [570,2340], [2590, 640], [2530,1820]])
pts2 = np.float32([[0,0], [0, height], [width, 0],[width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
homography = cv2.warpPerspective(src, matrix, (width, height))

rows = 2
cols = 2
fig = plt.figure(figsize=(8,8))
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

def cannyedge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize = 3)
    edges_true = np.where(edges == 255)
    print(edges_true)
    X_data = edges_true[0]
    Y_data = edges_true[1]
    return X_data, Y_data

x, y = cannyedge(homography)
ax3 = fig.add_subplot(rows,cols, 4)
ax3 = plt.scatter(x, y)
plt.show()

# from sklearn import linear_model
# def X_reshape(X):
#     X = np.reshape(X,(-1,1))
#     return X
# lr = linear_model.LinearRegression()
# lr.fit(X_reshape(X_data), Y_data)
# Y_predict = lr.predict(X_reshape(X_data),(-1,1))
#
# plt.scatter(X_data, Y_data,color='g')
# plt.plot(X_test, y_pred,color='k')