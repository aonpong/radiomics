import cv2
from scipy import ndimage, misc
import testGLCM
import numpy as np


def laplace_of_gaussian(img, sigmas):
    output = []
    for sigma in sigmas:
        output.append(ndimage.gaussian_laplace(img, sigma=sigma))
    return np.array(output)


if __name__ == '__main__':
    input_img = cv2.imread("D:\Capture.png", 0)
    reference = input_img.copy()
    cnt = 0

    sigma = [0, 1, 1.5, 2, 2.5]
    output = laplace_of_gaussian(input_img, sigma)

    cv2.imshow("Original", input_img)
    for i in range(len(sigma)):
        cv2.imshow("LoG" + str(sigma[i]), output[i])

    print("Pause for showing results")
    cv2.waitKey()

    print("End of Process")
