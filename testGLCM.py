from skimage.feature import greycomatrix
import numpy as np
import cv2


def image_cut(full_img, ref, x_start, x_end, y_start, y_end):
    draw = ref
    square = np.zeros((y_end-y_start, x_end-x_start), dtype=int)
    draw[y_start:y_end, x_start:x_end] = square
    return draw, full_img[y_start:y_end, x_start:x_end]


def image_slice(full_img, windows_size):
    full_img = np.array(full_img)
    row, col = full_img.shape
    print(row%windows_size, col%windows_size, row, windows_size)
    row_add = row + windows_size%row
    col_add = col + windows_size%col
    added_img = np.zeros((col_add, row_add), dtype=full_img.dtype)
    added_img[:, :] = np.mean(full_img)
    added_img[0:col, 0:row] = full_img[:, :]
    #cv2.imshow("Added", added_img)
    #cv2.imshow("Full", full_img)
    #cv2.waitKey()

    sliced = []
    animate = []
    for i in range (0, row, windows_size):
        for j in range (0, col, windows_size):
            #sliced.append(full_img[i: min(i+windows_size, row), j: min(j+windows_size, col)])
            sliced.append(added_img[i: i + windows_size, j: j + windows_size])
            showblock = added_img.copy()
            showblock[i: i+windows_size, j: j+windows_size] = 255
            animate.append(showblock)
            #cv2.imshow("show", showblock)
            #cv2.imshow("show_subblock", added_img[i: i + windows_size, j: j + windows_size])
            #print np.array(sliced).shape
            #cv2.waitKey(50)

    print("shape sliced", np.array(sliced).shape)
    print(len(sliced))
    print(sliced, added_img.dtype)

    return sliced, np.array(animate)


def glcm4direction(img, elements):
    result_glcm = np.array(greycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=max(elements) + 1)).astype(
        np.uint8)
    #result_glcm = result_glcm/np.max(result_glcm)
    #print "GLCM: ", result_glcm.shape
    return result_glcm[:, :, 0, 0], result_glcm[:, :, 0, 1], result_glcm[:, :, 0, 2], result_glcm[:, :, 0, 3], result_glcm


if __name__ == '__main__':
    input_img = cv2.imread("Lenna.png", 0)
    reference = input_img.copy()

    element = list(np.unique(np.array(input_img)))
    sliced, animate = image_slice(input_img, 40)

    for i in range(len(sliced)):
        output0, output90, output180, output270 = glcm4direction(sliced[i], element)
        #output0 = cv2.resize(output0, (256,256))
        #output90 = cv2.resize(output90, (256,256))
        #output180 = cv2.resize(output180, (256,256))
        #output270 = cv2.resize(output270, (256,256))
        cv2.imshow("ShowBlock", np.float32(animate[i])/np.float32(animate[i].max()))
        cv2.imshow("GLCM_0", output0)
        cv2.imshow("GLCM_90", output90)
        cv2.imshow("GLCM_180", output180)
        cv2.imshow("GLCM_270", output270)
        if i == 0:
            print("Press any key to start")
            cv2.waitKey()
        else:
            cv2.waitKey(200)

    exit()

    reference, cut = image_cut(input_img, reference, 20, 400, 20, 400)

    output0, output90, output180, output270 = glcm4direction(cut, element)

    cv2.imshow("Full image", input_img)

    cv2.imshow("Cut", cut)
    cv2.imshow("Drw", reference)

    cv2.imshow("GLCM_0", output0)
    cv2.imshow("GLCM_90", output90)
    cv2.imshow("GLCM_180", output180)
    cv2.imshow("GLCM_270", output270)

    print("GLCM has successfully operated")

    cv2.waitKey()
    print("End of process")
