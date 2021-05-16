import SimpleITK as sitk
import numpy as np
import cv2
import os
import pydicom as ds

#dcm = ds.read_file(r'D:\1420dataset\test\R01-003\\')
#a = dcm.pixel_array
#print(a.shape)
#
#exit()
# --------------------------- NII Reader ------------------------------------
r_01 = []
DCM = True
if DCM is True:
    path = r'D:\1420dataset\screened\\' #603-CT Thick Axials 2.5mm-32453\\'
    cnt = 0
    for r, d, f in os.walk(path):
        for directory in d:
            print(directory)
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path + directory)
            reader.SetFileNames(dicom_names)

            image = reader.Execute()
            size = image.GetSize()
            t1 = np.array(sitk.GetArrayFromImage(image))
            t1[t1 == -2000] = 0
            #print(t1.shape)


            nrrd = sitk.ReadImage(path + directory + r"\\" + "1.nrrd")
            t2 = sitk.GetArrayFromImage(nrrd)

            full_sample = np.array(t1)
            t1 = None

            MIN_BOUND = np.float32(-1000) #-1000.0
            MAX_BOUND = np.float32(400) #400.0
            image = (full_sample - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
            image[image > 1] = 1.
            image[image < 0] = 0.
            norm_sample = image*255

            # --------------------------- NRRD Reader ------------------------------------

            NRRD = True
            if NRRD is True:

                '''MIN_BOUND = np.float32(full_sample.min()) #-1000.0
                MAX_BOUND = np.float32(full_sample.max()) #400.0
                image = (full_sample - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
                image[image > 1] = 1.
                image[image < 0] = 0.'''

            if full_sample.shape[0] == t2.shape[0]:
                sum_list = []
                for i in range(full_sample.shape[0]):
                    sum_list.append(t2[i, :, :].sum())

                sum_list = np.array(sum_list)

                for j in range(len(sum_list)):
                    if sum_list[j] == sum_list.max():
                        z = [j-1, j, j+1]

                x_min_list = []
                x_max_list = []
                y_min_list = []
                y_max_list = []

                img_removed_edge = np.zeros((3, t2.shape[1], t2.shape[2]))
                for i in range(3):
                    img_removed_edge[i, :, :] = t2[z[i], :, :] * norm_sample[z[i], :, :]

                for i in z:
                    for j in range(512):
                        if t2[i, j, :].sum() > 0:
                            y_max = j
                    y_max_list.append(y_max)

                    for j in reversed(range(512)):
                        if t2[i, j, :].sum() > 0:
                            y_min = j
                    y_min_list.append(y_min)

                    for j in range(512):
                        if t2[i, :, j].sum() > 0:
                            x_max = j
                    x_max_list.append(x_max)

                    for j in reversed(range(512)):
                        if t2[i, :, j].sum() > 0:
                            x_min = j
                    x_min_list.append(x_min)

                x_min_list = np.array(x_min_list)
                x_max_list = np.array(x_max_list)
                y_min_list = np.array(y_min_list)
                y_max_list = np.array(y_max_list)

                for i in range(3):
                    if x_min_list.min() == x_min_list[i]:
                        x_min = x_min_list[i]
                    if x_max_list.max() == x_max_list[i]:
                        x_max = x_max_list[i]
                    if y_min_list.min() == y_min_list[i]:
                        y_min = y_min_list[i]
                    if y_max_list.max() == y_max_list[i]:
                        y_max = y_max_list[i]

                print("all:", x_min_list, x_max_list, y_min_list, y_max_list)
                print("chs:", x_min, x_max, y_min, y_max, z[1])

                for j in range(512):
                    image[i, y_min, j] = 1
                    image[i, y_max, j] = 1
                    image[i, j, x_min] = 1
                    image[i, j, x_max] = 1


                #tumor = cv2.resize(norm_sample[z[1], y_min:y_max, x_min:x_max], (224, 224))
                #cv2.imshow("TEST_CUT", cv2.resize(img_removed_edge[1, y_min:y_max, x_min:x_max], (224, 224)))
                #cv2.imshow("TEST_FULL", cv2.resize(norm_sample[z[1], :, :], (224, 224)))

                set = np.zeros((224, 224, 3))
                for i in range(3):
                    set[:, :, i] = cv2.resize(img_removed_edge[i, y_min:y_max, x_min:x_max], (224, 224))

            else:
                print("Size not match. Using initial set. :", full_sample.shape[0], t2.shape[0])

                sum_list = []
                initial = [120, 219, 156, 195, 138, 203, 28, 95, 206, 146, 71, 45, 84, 53, 150, 91, 103, 154, 97, 82, 166, 87, 177, 73, 157, 152, 196, 194, 61, 193, 72, 237, 88, 99, 226, 60, 201, 148, 144, 155, 57, 143, 157, 134, 97]
                      # tmp 22,  26, 101, 102, 103, 104,105,106, 107, 108,109,110,111,112, 113,114, 115, 116,117,118, 120,121, 122,123, 124, 125, 126, 127,128, 129,130, 131,132,134, 135, 136,137, 138, 139, 140,141, 142, 144, 145,146
                for i in range(t2.shape[0]):
                    sum_list.append(t2[i, :, :].sum())

                sum_list = np.array(sum_list)

                for j in range(len(sum_list)):
                    if sum_list[j] == sum_list.max():
                        z = [j - 1, j, j + 1]

                x_min_list = []
                x_max_list = []
                y_min_list = []
                y_max_list = []

                img_removed_edge = np.zeros((3, t2.shape[1], t2.shape[2]))
                for i in range(3):
                    img_removed_edge[i, :, :] = t2[z[i], :, :] * norm_sample[z[i] + initial[cnt], :, :]
                for i in z:
                    for j in range(512):
                        if t2[i, j, :].sum() > 0:
                            y_max = j
                    y_max_list.append(y_max)

                    for j in reversed(range(512)):
                        if t2[i, j, :].sum() > 0:
                            y_min = j
                    y_min_list.append(y_min)

                    for j in range(512):
                        if t2[i, :, j].sum() > 0:
                            x_max = j
                    x_max_list.append(x_max)

                    for j in reversed(range(512)):
                        if t2[i, :, j].sum() > 0:
                            x_min = j
                    x_min_list.append(x_min)

                x_min_list = np.array(x_min_list)
                x_max_list = np.array(x_max_list)
                y_min_list = np.array(y_min_list)
                y_max_list = np.array(y_max_list)

                for i in range(3):
                    if x_min_list.min() == x_min_list[i]:
                        x_min = x_min_list[i]
                    if x_max_list.max() == x_max_list[i]:
                        x_max = x_max_list[i]
                    if y_min_list.min() == y_min_list[i]:
                        y_min = y_min_list[i]
                    if y_max_list.max() == y_max_list[i]:
                        y_max = y_max_list[i]

                print("all:", x_min_list, x_max_list, y_min_list, y_max_list)
                print("chs:", x_min, x_max, y_min, y_max, z[1], z[1] + initial[cnt])

                for j in range(512):
                    image[i + initial[cnt], y_min, j] = 1
                    image[i + initial[cnt], y_max, j] = 1
                    image[i + initial[cnt], j, x_min] = 1
                    image[i + initial[cnt], j, x_max] = 1

                #tumor = cv2.resize(img_removed_edge[z[1], y_min:y_max, x_min:x_max], (224, 224))
                #cv2.imshow("TEST_CUT", cv2.resize(norm_sample[z[1] + initial[cnt], y_min:y_max, x_min:x_max], (224, 224)))
                #cv2.imshow("TEST_FULL", cv2.resize(norm_sample[z[1] + initial[cnt], :, :], (224, 224)))

                #cv2.waitKey()

                cnt += 1

                set = np.zeros((224, 224, 3))
                for i in range(3):
                    set[:, :, i] = cv2.resize(img_removed_edge[i, y_min:y_max, x_min:x_max], (224, 224))

            r_01.append(set)
            print(r"Writing to C:\Users\panya\Desktop\R01_cut\R01_" + str(directory[-3:] + ".jpg"), np.array(set).shape)
            cv2.imwrite(r"C:\Users\panya\Desktop\R01_cut\R01_" + str(directory[-3:] + ".jpg"), np.array(set))

    r_01 = np.array(r_01)
    print("Data set was saved into NumPy array as shape :", r_01.shape)
    np.save(r"dataset\\ds_test.npy", r_01)
    print("Wrote all files successful")
