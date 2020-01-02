import torch
import torch.nn as nn


def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (excluding exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    # l_mu = [1]
    # if n_kernels == 1:
    #     return l_mu
    l_mu = list()

    bin_size = 1.8 / n_kernels  # score range from [-0.9, 0.9]
    l_mu.append(0.9 - bin_size / 2)  # mu: middle of the bin
    for i in range(n_kernels):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (excluding exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of sigma
    """
    # l_sigma = [0.001]  # for exact match. small variance -> exact match
    # if n_kernels == 1:
    #     return l_sigma

    l_sigma = [0.5] * n_kernels
    return l_sigma


def get_intersect_matrix(qry, psg):
    q_len = qry.shape[1]
    p_len = psg.shape[1]
    bs = qry.shape[0]
    qry = qry.permute(1, 0, 2)
    psg = psg.permute(1, 0, 2)
    inter_matrix = torch.zeros([q_len, p_len, bs])
    for i in range(q_len):
        for j in range(p_len):
            inter_matrix[i][j] = torch.cosine_similarity(qry[i], psg[j], dim=1)

    return inter_matrix.permute(2, 0, 1)


class KNRM(nn.Module):
    def __init__(self, batch_size, kernal_num):
        super(KNRM, self).__init__()
        self.kernal_num = kernal_num
        self.proj_match = nn.Linear(4 * kernal_num, 1, bias=True)
        self.bn = nn.BatchNorm1d(4 * self.kernal_num)
        self.batch_size = batch_size

    def forward(self, qry_wrd, psg_wrd, qry_ent, psg_ent):
        q_len = 15
        q_ent_len = qry_ent.shape[1]
        inter_ww = get_intersect_matrix(qry_wrd, psg_wrd).cuda()  # size: BS * query_len * passage_len
        inter_we = get_intersect_matrix(qry_wrd, psg_ent).cuda()
        inter_ew = get_intersect_matrix(qry_ent, psg_wrd).cuda()
        inter_ee = get_intersect_matrix(qry_ent, psg_ent).cuda()
        mus = kernal_mus(self.kernal_num)
        sigmas = kernel_sigmas(self.kernal_num)

        # TODO: check computation
        K_ww = torch.zeros(self.batch_size, self.kernal_num, q_len).cuda()  # size: BS * K * query_len
        K_we = torch.zeros(self.batch_size, self.kernal_num, q_len).cuda()
        K_ew = torch.zeros(self.batch_size, self.kernal_num, q_ent_len).cuda()
        K_ee = torch.zeros(self.batch_size, self.kernal_num, q_ent_len).cuda()
        for i in range(self.batch_size):
            for j in range(self.kernal_num):
                for k in range(q_len):
                    K_ww[i][j][k] = torch.sum(torch.exp(-(inter_ww[i][k] - mus[j]) ** 2 / (2 * sigmas[j] ** 2)), dim=0)
                    K_we[i][j][k] = torch.sum(torch.exp(-(inter_we[i][k] - mus[j]) ** 2 / (2 * sigmas[j] ** 2)), dim=0)
                    if k < q_ent_len:
                        K_ew[i][j][k] = torch.sum(torch.exp(-(inter_ew[i][k] - mus[j]) ** 2 / (2 * sigmas[j] ** 2)),
                                                  dim=0)
                        K_ee[i][j][k] = torch.sum(torch.exp(-(inter_ee[i][k] - mus[j]) ** 2 / (2 * sigmas[j] ** 2)),
                                                  dim=0)
        phi_ww = torch.sum(torch.log(K_ww), dim=2)  # size: BS * K
        phi_we = torch.sum(torch.log(K_we), dim=2)
        phi_ew = torch.sum(torch.log(K_ew), dim=2)
        phi_ee = torch.sum(torch.log(K_ee), dim=2)
        phi = torch.cat([phi_ww, phi_we, phi_ew, phi_ee], dim=1)
        phi = self.bn(phi)
        score1 = torch.tanh(self.proj_match(phi).squeeze())
        return score1