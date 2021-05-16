from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import csv

def lasso(training, gtruth, predict=None, answer=None, alpha = 0.0001):
    clf0 = None
    clf0 = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=1000,
                              normalize=False, positive=False, precompute=False, random_state=None,
                              selection='cyclic', tol=0.0001, warm_start=False)

    clf0.fit(training, gtruth)
    if predict is not None:
        result = clf0.predict(predict)
        middle = 0.5#(max(result) + min(result))/2

        for i in range(len(result)):
            #print result[i]
            if result[i] > middle:
                result[i] = 1
            else:
                result[i] = 0
            #print "AFTER :", result[i]

        acc = roc_auc_score(answer, result)
        cc = []
        for i in range(len(clf0.coef_)):
            cc.append(bool(clf0.coef_[i]))

        return result, clf0, clf0.coef_, acc
    else:
        return "No predict set", clf0,clf0.coef_

if __name__ == '__main__':

    rna_data = np.swapaxes(np.load(r"dataset\RNA_seq_recurrence.npy"), 0, 1).astype(np.float64)
    gt = np.load(r"dataset\recurrence_gt.npy")
    rna_name_list = np.load(r"dataset\RNA_seq_name.npy")
    print(rna_data.shape, gt.shape)
    #for i in feats_data:
    #    print(i)
    #print(feats_data)
    active_table = np.zeros((3, rna_data.shape[1]))

    # ---------------------------------------------- LASSO ----------------------------------------------
    print("--------------------- LASSO -----------------------")
    result, clf0, coef = lasso(rna_data, gt)
    count = 1
    tmp = []
    tmp_name = []
    for i in range(len(coef)):
        if np.abs(coef[i]) > 0.0:
            print(count, '\t', rna_name_list[i], '\t', coef[i])
            active_table[0, i] = 1
            count += 1
            tmp.append(rna_data[:, i])
            tmp_name.append(rna_name_list[i])
    # ----------------------------------------------- ANOVA -------------------------------------------
    print("--------------------- ANOVA -----------------------")
    print(rna_data.shape, np.array(tmp).shape)
    tmp = np.swapaxes(np.array(tmp), 0, 1)
    print(tmp.shape)
    selector1 = SelectKBest(f_classif, k=200)
    selector1.fit(tmp, gt)
    count = 1

    tmp3 = []
    tmp_name3 = []
    tmp_p3 = []
    for i in range(len(selector1.pvalues_)):
        if selector1.pvalues_[i] < 0.05:
            print(count, '\t', selector1.pvalues_[i], '\t', tmp_name[i])
            active_table[1, i] = 1
            count += 1
            tmp3.append(rna_data[:, i])
            tmp_p3.append(selector1.pvalues_[i])
            tmp_name3.append(tmp_name[i])
    arr1inds = np.argsort(tmp_p3)
    tmp3 = np.array(tmp3)
    print(tmp.shape)
    tmp2 = []
    tmp_p2 = []
    tmp_name2 = []
    print(arr1inds)
    for i in range(len(tmp_p3)):
        tmp2.append(tmp3[arr1inds[i]])
        tmp_p2.append(tmp_p3[arr1inds[i]])
        tmp_name2.append(tmp_name3[arr1inds[i]])
        print(i+1, ' ', tmp_p2[i], ' ', tmp_name2[i])

    np.save(r"dataset\rank_Feats_recurrence_anova.npy", tmp2)

    # ----------------------------------------------- CHI2 -------------------------------------------

    writer = csv.writer(open(r"dataset\Rank_Feats_active2.csv", 'w'))
    writer.writerow(rna_name_list)
    for row in active_table:
        writer.writerow(row)
