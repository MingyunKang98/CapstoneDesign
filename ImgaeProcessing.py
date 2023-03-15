import cv2
import matplotlib.pyplot as plt
import numpy as np

dir = "./2.jpg"
src = cv2.imread(dir)

print(np.shape(src))
def homography(src):
    width = np.shape(src)[0]
    height = np.shape(src)[1]
    pts1 = np.float32([[460,310], [570,2340], [2590, 640], [2530,1820]])
    pts2 = np.float32([[0,0], [0, height], [width, 0],[width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    homography = cv2.warpPerspective(src, matrix, (width, height))
    return homography

def cannyedge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 100, apertureSize = 3)
    edges_true = np.where(edges == 255)
    print(edges_true)
    X_data = edges_true[0]
    Y_data = edges_true[1]
    return X_data, Y_data

if __name__ == "__main__":
    x, y = cannyedge(homography(src))
    from ransac_wiki import RANSAC, LinearRegressor, square_error_loss, mean_square_error
    regressor = RANSAC(model=LinearRegressor(), loss=square_error_loss, metric=mean_square_error)

    ransac_X = np.array(x).reshape(-1,1)
    ransac_y = np.array(y).reshape(-1,1)

    regressor.fit(ransac_X, ransac_y)

    import matplotlib.pyplot as plt
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots(1, 1)
    ax.set_box_aspect(1)

    # plt.scatter(ransac_X, ransac_y)

    line = np.linspace(-1, 1, num=100).reshape(-1, 1)
    plt.plot(line, regressor.predict(line), c="peru")
    plt.show()