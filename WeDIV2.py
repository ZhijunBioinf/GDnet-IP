import numpy as np
from scipy.stats import zscore
from scipy.stats import f
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def similarity_euclid(data):  # Calculate similarity matrix
    Z = pdist(data, 'euclidean')
    Edist = squareform(Z)
    return Edist


def nornaliz(data):  # Normalize one row or one column with zeros in the diagonal
    nrow = data.shape[0]
    M = data.copy()
    np.fill_diagonal(M, 0)
    dmax = np.max(M)
    np.fill_diagonal(M, 5)
    dmin = np.min(M)
    cmax = dmax - dmin
    M = M - np.tile(dmin, (nrow, 1))
    M = M / np.tile(cmax, (nrow, 1))
    np.fill_diagonal(M, 0)
    return M


def nornaliz_1(data):  # Normalize the matrix
    M = (data - np.min(data)) / (np.max(data) - np.min(data))
    return M


def my_anova1(X, Y):  # One way ANOVA
    labels = np.unique(Y)
    if len(labels) == 1:
        F = 0
        s2t = 0
        s2e = 0
        return F, s2t, s2e

    class_num = len(labels)
    Ti = np.zeros(class_num)
    ni = np.zeros(class_num)
    for m in range(class_num):
        Ti[m] = np.sum(X[Y == labels[m]])
        ni[m] = np.sum(Y == labels[m])

    C = (np.sum(Ti) ** 2) / np.sum(ni)
    SST = np.sum(X ** 2) - C
    SSt = np.sum(Ti ** 2 / ni) - C
    SSe = SST - SSt
    dfT = len(X) - 1
    dft = class_num - 1
    dfe = dfT - dft
    s2t = SSt / dft
    s2e = SSe / dfe
    F = s2t / s2e
    return F, s2t, s2e


def get_CH(Sdata, fy):  # Calculate the RCH value
    mcol = Sdata.shape[1]
    F = np.zeros(mcol)
    Mst = np.zeros(mcol)
    Mse = np.zeros(mcol)
    for i in range(mcol):
        F[i], Mst[i], Mse[i] = my_anova1(Sdata[:, i], fy)
    CH = np.sum(Mst) / np.sum(Mse)
    CH_0 = np.sum(F)
    return CH, CH_0


def clustering(Sdata, C, w): # K-means like clustering invoked by the main clustering function
    """
    :param Sdata: normalized initial data
    :param C: centroids
    :param w: distance weight
    :return: clustering labels for each sample
    """
    nrow = Sdata.shape[0]
    K = C.shape[0]
    Sdata = Sdata.T
    C = C.T
    Edist_c = np.zeros((nrow, K))
    R_c = Edist_c.copy()

    for j in range(K):
        for i in range(nrow):
            Edist_c[i, j] = np.linalg.norm(Sdata[:, i] - C[:, j])
            r = np.corrcoef(Sdata[:, i], C[:, j])
            if np.isnan(r[0, 1]):
                R_c[i, j] = 0
            else:
                R_c[i, j] = r[0, 1]
    Edist_c0 = nornaliz_1(Edist_c)
    R_c0 = nornaliz_1(R_c)

    if np.all(np.isnan(R_c0)):
        ED_Pearson_1 = Edist_c0
    else:
        ED_Pearson_1 = w * Edist_c0 + (1 - w) * (1 - R_c0)
    fy = np.argmin(ED_Pearson_1, axis=1)
    return fy


def optimizeWeight(Sdata, E, R, K, w_step): # Optimize the distance weight
    nrow, mcol = Sdata.shape
    n_w = 1
    num_step = int(1 / w_step) + 1
    CH_W_Y = np.zeros((nrow, num_step))
    CH_W = np.full((1, num_step), np.nan).T
    CH_W_0 = CH_W.copy()

    for w in np.arange(0, 1 + w_step, w_step):
        ED_Pearson = w * E + (1 - w) * (1 - R)
        np.fill_diagonal(ED_Pearson, 0)
        c1, c2 = np.unravel_index(np.argmax(ED_Pearson), ED_Pearson.shape)
        center = [c2]
        center.append(c1)

        for v in range(2, K):
            dis0 = ED_Pearson[center, :]
            d3 = np.argmax(np.min(dis0, axis=0))
            center.append(d3)
        dis = ED_Pearson[center, :]
        fy = np.argmin(dis, axis=0)
        count = 0
        C = Sdata[center, :].copy()
        C_adjust = np.zeros((K, mcol))

        while count < 50 and not np.array_equal(C, C_adjust):
            C_adjust = C.copy()
            for i in range(K):
                Cindex = np.where(fy == i)[0]
                n_i = len(Cindex)
                if n_i > 1:
                    data_i = Sdata[Cindex, :]
                    ED_i = ED_Pearson[np.ix_(Cindex, Cindex)]
                    lab = np.argmin(np.sum(ED_i, axis=1))
                    C[i, :] = data_i[lab, :]
                    center[i] = lab
            fy = clustering(Sdata, C, w)
            count += 1
        CH_W[n_w - 1], CH_W_0[n_w - 1] = get_CH(Sdata, fy)
        CH_W_Y[:, n_w - 1] = fy
        n_w += 1

    CH_Wmax = np.max(CH_W)
    lab = np.argmax(CH_W)
    RCH_K = CH_Wmax / f.ppf(0.99, K - 1, nrow - K)
    RCH_K_Y = CH_W_Y[:, lab]
    w_optimal = lab * 0.1

    return RCH_K_Y, RCH_K, w_optimal


def WeDIV2(data, w_step, KList): # Main function of WeDIV clustering
    """
    WeDIV: clustering with Weighted Distance and novel Internal Validation index
    :param data: a matrix, original dataset
    :param w_step: step of distance weight to be optimized (only use Euclidean distance when setting as 1)
    :param KList: a vector, range of the number of clustering
    :return: Y_CL, final clustering labels for each sample
    :return: K_optimal, optimal clustering number
    :return: W_optimal, optimal distance weight between Pearson correlation and Euclidean distances
    """
    nrow = data.shape[0]
    Sdata = zscore(data, ddof=1)
    E0 = similarity_euclid(Sdata)
    E1 = nornaliz(E0)
    R0 = np.corrcoef(Sdata)
    R1 = nornaliz(R0)

    RCH_K_Y = np.zeros((nrow, len(KList)))
    RCH_K = np.zeros(len(KList))
    w_optimal = np.zeros(len(KList))
    for k in KList:
        nk = 0
        RCH_K_Y[:, nk], RCH_K[nk], w_optimal[nk] = optimizeWeight(Sdata, E1, R1, k, w_step)
    nk = nk + 1
    lab = np.argmax(RCH_K)
    max_RCH = max(RCH_K)
    if max(RCH_K) < 1:
        Y_CL = np.full(nrow, 1)
        K_optimal = 1
        W_optimal = np.nan
    else:
        K_optimal = KList[lab]
        Y_CL = RCH_K_Y[:, lab] + 1
        W_optimal = w_optimal[lab]

    return Y_CL, K_optimal, W_optimal, max_RCH


if __name__ == '__main__':
    Qdata = pd.read_excel(r'/path/to/your/data.xlsx', header=None)
    Y_CL, optiKCluster, W_optimal = WeDIV2(Qdata.iloc[:, 1:].to_numpy(), 0.1, range(2, 9))
