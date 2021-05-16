from sklearn import linear_model
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import csv

def lasso(training, gtruth, predict=None, answer=None, alpha = 0.001):
    clf0 = None
    clf0 = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=5000,
                              normalize=False, positive=False, precompute=False, random_state=None,
                              selection='cyclic', tol=0.019443181818181818, warm_start=False)

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

    feats_data = np.abs(np.load(r"dataset\Feats_recurrence_new.npy", allow_pickle=True).astype(np.float64))
    gt = np.load(r"dataset\recurrence_gt.npy")
    #rna_name_list = np.load(r"dataset\RNA_seq_name.npy")
    print(feats_data.shape, gt.shape)
    for i in feats_data:
        print(i)
    #print(feats_data)
    active_table = np.zeros((3, feats_data.shape[1]))

    # ---------------------------------------------- LASSO ----------------------------------------------
    print("--------------------- LASSO -----------------------")
    result, clf0, coef = lasso(np.abs(feats_data), gt)
    count = 1
    tmp = []
    tmp_name = []
    for i in range(len(coef)):
        if np.abs(coef[i]) > 0.00001:
            print(count, '\t', coef[i])
            active_table[0, i] = 1
            count += 1
            tmp.append(feats_data[:, i])

    #np.save(r"dataset\new_Feats_recurrence_lasso.npy", tmp)

    # ----------------------------------------------- ANOVA -------------------------------------------
    print("--------------------- ANOVA -----------------------")
    selector1 = SelectKBest(f_classif, k=200)
    selector1.fit(np.abs(feats_data), gt)
    count = 1

    tmp = []
    tmp_name = []
    for i in range(len(selector1.pvalues_)):
        if selector1.pvalues_[i] < 0.05:
            print(count, '\t', selector1.pvalues_[i])
            active_table[1, i] = 1
            count += 1
            tmp.append(feats_data[:, i])
            #tmp_name.append(rna_name_list[i])
    #np.save(r"dataset\new_Feats_recurrence_anova.npy", tmp)

    # ----------------------------------------------- CHI2 -------------------------------------------
    print("--------------------- CHI2 -----------------------")
    selector2 = SelectKBest(chi2, k=4)
    selector2.fit(np.abs(feats_data), gt)
    count = 1

    tmp = []
    tmp_name = []
    for i in range(len(selector2.pvalues_)):
        if selector2.pvalues_[i] < 0.05:
            print(count, '\t', selector2.pvalues_[i])
            active_table[2, i] = 1
            count += 1
            tmp.append(feats_data[:, i])
    #np.save(r"dataset\new_Feats_recurrence_chi2.npy", tmp)


    #writer = csv.writer(open("dataset\Feats_active2.csv", 'w'))
    #writer.writerow(rna_name_list)
    #for row in active_table:
    #    writer.writerow(row)
