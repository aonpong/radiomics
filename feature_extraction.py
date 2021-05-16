import SimpleITK as sitk
import numpy as np
import testGLCM
import testLoG
import scipy.stats
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import feature
from xlrd import open_workbook
import cv2, os, math

def LGauss(image):
    print("Start LGauss...")
    image_set = image
    gauss = testLoG.laplace_of_gaussian(image_set - image_set.min()-1, [0, 1, 1.5, 2, 2.5])
    min_g = np.uint8(np.array(gauss).min()-1)

    gauss -= min_g

    min_g = None
    glcm4way = []
    coprobs = []
    for i in gauss:
        # print i.shape, i.max(), i.min()
        tmp = testGLCM.glcm4direction(i, np.unique(i))

        entr_find1 = np.float16(tmp[0]) / np.float16(np.max(tmp[0]))
        entr_find2 = np.float16(tmp[1]) / np.float16(np.max(tmp[1]))
        entr_find3 = np.float16(tmp[2]) / np.float16(np.max(tmp[2]))
        entr_find4 = np.float16(tmp[3]) / np.float16(np.max(tmp[3]))
        coprob_find = np.float16(tmp[4])/np.float16(np.max(tmp[4]))
        entr_img1 = entropy(entr_find1, disk(10)).sum()
        entr_img2 = entropy(entr_find2, disk(10)).sum()
        entr_img3 = entropy(entr_find3, disk(10)).sum()
        entr_img4 = entropy(entr_find4, disk(10)).sum()
        entr_img = np.array([entr_img1, entr_img2, entr_img3, entr_img4])
        contrast = np.array(feature.texture.greycoprops(coprob_find, prop='contrast'))
        homo = np.array(feature.texture.greycoprops(coprob_find, prop='homogeneity'))
        energy = np.array(feature.texture.greycoprops(coprob_find, prop='energy'))
        corr = np.array(feature.texture.greycoprops(coprob_find, prop='correlation'))
        glcm4way.append(tmp[0:4])
        tmp = np.append(contrast, corr)
        tmp = np.append(tmp, entr_img)
        tmp = np.append(tmp, energy)
        tmp = np.append(tmp, homo)
        coprobs = np.append(coprobs, tmp)
        tmp = None

    print(contrast, corr, entr_img, energy, homo)
    return glcm4way, np.array(coprobs)

def histo(image):
    print("Start histo...")
    image_set = image
    gauss = testLoG.laplace_of_gaussian(image_set - image_set.min(), [0, 1, 1.5, 2, 2.5])
    min_g = np.uint8(np.array(gauss).min())
    gauss -= min_g
    result = []

    for a in gauss:
        a = (np.array(a)).reshape((a.shape[0] * a.shape[1], 1))
        a_mean = np.mean(a)
        a_std = np.std(a)
        a_kurtosis = scipy.stats.kurtosis(a)
        a_skew = scipy.stats.skew(a)
        pct10 = np.percentile(a, 10)
        pct25 = np.percentile(a, 25)
        pct50 = np.percentile(a, 50)
        pt10 = []
        pt25 = []
        pt50 = []

        for i in a:
            if i < pct50:
                pt10.append(i)
                pt25.append(i)
                pt50.append(i)
            elif i < pct25:
                pt10.append(i)
                pt25.append(i)
            elif i < pct10:
                pt10.append(i)
        a_mean10 = np.mean(pt10)
        a_mean25 = np.mean(pt25)
        a_mean50 = np.mean(pt50)
        a_sd10 = np.std(pt10)
        a_sd25 = np.std(pt25)
        a_sd50 = np.std(pt50)
        #result.append(np.append(np.array([a_mean, a_std, a_kurtosis, a_skew, a_mean10, a_mean25, a_mean50, a_sd10, a_sd25, a_sd50]), []))
        if math.isnan(a_mean) is True:
            a_mean = 0.
        result.append(a_mean)
        if math.isnan(a_std) is True:
            a_std = 0.
        result.append(a_std)
        if math.isnan(a_kurtosis) is True:
            a_kurtosis = 0.
        result.append(a_kurtosis)
        if math.isnan(a_skew) is True:
            a_skew = 0.
        result.append(a_skew)
        if math.isnan(a_mean10) is True:
            a_mean10 = 0.
        result.append(a_mean10)
        if math.isnan(a_mean25) is True:
            a_mean25 = 0.
        result.append(a_mean25)
        if math.isnan(a_mean50) is True:
            a_mean50 = 0.
        result.append(a_mean50)
        if math.isnan(a_sd10) is True:
            a_sd10 = 0.
        result.append(a_sd10)
        if math.isnan(a_sd25) is True:
            a_sd25 = 0.
        result.append(a_sd25)
        if math.isnan(a_sd50) is True:
            a_sd50 = 0.
        result.append(a_sd50)

    print(result)
    return np.array(result)


def merge_features(histo, greyco):
    return np.append(histo, greyco)


def main(target_folder):
    feats = []
    for r, d, f in os.walk(os.path.join(target_folder)):
        for file in f:
            print(os.path.join(target_folder, file))
            a1 = cv2.imread(os.path.join(target_folder, file))
            print(a1.shape)

            print("Working with ", file)
            GLCM54_a, coprob_a = LGauss(a1[:, :, 0])
            res_a = histo(a1[:, :, 0])
            feat_a = merge_features(res_a, coprob_a)

            GLCM54_b, coprob_b = LGauss(a1[:, :, 1])
            res_b = histo(a1[:, :, 1])
            feat_b = merge_features(res_b, coprob_b)

            GLCM54_c, coprob_c = LGauss(a1[:, :, 2])
            res_c = histo(a1[:, :, 2])
            feat_c = merge_features(res_c, coprob_c)

            feat = np.array([feat_a, feat_b, feat_c])
            #print(feat)
            print(feat.shape)
            feat = feat.ravel()
            print(feat)
            feats.append(feat)
        print(len(feats), len(feats[0]), len(feats[1]))


        np.save(r"dataset\feats_recurrence_new.npy", np.array(feats))
        print("Files saved as dimension :", np.array(feats).shape)


if __name__ == '__main__':

    target_folder = r"C:\Users\panya\Desktop\R01_recur_88"
    main(target_folder)
