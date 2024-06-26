import numpy as np
import torch
import itertools
from utils.util import tensor2np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class Maxlloyd:
    def __init__(self, scores, rpt_num):
        self.scores = scores
        self.scores[self.scores == 0] = self.scores[self.scores.argsort()][1]

        score_min = self.scores.min()
        score_max = self.scores.max()

        rpt_scores = np.arange(start=score_min, stop=score_max, step=np.abs(score_min - score_max) / (rpt_num - 2))
        self.init_r = np.array([0] + rpt_scores.tolist() + [100])
        self.init_t = np.array([((self.init_r[r] + self.init_r[r + 1]) / 2) for r in range(len(self.init_r) - 1)])

    def get_new_rpt_scores(self):

        prev_r = self.init_r
        prev_t = self.init_t

        while True:
            new_r = self.update_r(prev_r, prev_t)
            new_t = self.update_t(new_r)

            if ((prev_t == new_t).sum() == len(new_t)) | ((prev_r == new_r).sum() == len(new_r)):
                break
            else:
                prev_r = new_r
                prev_t = new_t

        self.compute_q_error(np.round_(self.init_r, 5), np.round_(new_r, 5))
        self.get_memeber_num(np.round_(new_r, 2), self.scores)

        return np.round_(new_r, 2)
    def compute_q_error(self, init_r, new_r):
        argmin_init_r = cdist(self.scores.reshape(-1, 1), init_r.reshape(-1, 1)).argmin(1)
        argmin_new_r = cdist(self.scores.reshape(-1, 1), new_r.reshape(-1, 1)).argmin(1)

        q_error_init_r = np.abs(self.scores - init_r[argmin_init_r]).mean()
        q_error_new_r = np.abs(self.scores - new_r[argmin_new_r]).mean()

        print(f'Q eror | Init R: {q_error_init_r:.5f} | New R: {q_error_new_r:.5f}')

    def update_r(self, r, t):

        def strictly_increasing(L):
            return all(x < y for x, y in zip(L, L[1:]))

        def strictly_decreasing(L):
            return all(x > y for x, y in zip(L, L[1:]))

        def strictly_monotonic(L):
            return strictly_increasing(L) or strictly_decreasing(L)

        new_r = []
        new_r.append(r.min())
        for idx in range(len(t) - 1):
            selected_idxs = np.where((self.scores >= t[idx]) & (self.scores < t[idx + 1]))[0]
            if len(selected_idxs) > 0:
                mean = self.scores[selected_idxs].mean()
                #new_r.append(np.round_(mean, 2))
                new_r.append(mean)
            else:
                new_r.append(-1)
        new_r.append(r.max())

        new_r = np.array(new_r)
        invalid_r_idxs = np.where(new_r == -1)[0]

        if len(invalid_r_idxs) > 0:
            for invalid_idx in invalid_r_idxs:
                valid_r_idxs = np.where(new_r != -1)[0]

                dist_w_valid = cdist(invalid_idx.reshape(-1, 1), valid_r_idxs.reshape(-1, 1)).squeeze()
                dist_argsort = np.argsort(dist_w_valid)
                valid_idxs_argsort = valid_r_idxs[dist_argsort]

                prev = valid_idxs_argsort[np.where(valid_idxs_argsort < invalid_idx)[0][0]]
                next = valid_idxs_argsort[np.where(valid_idxs_argsort > invalid_idx)[0][0]]

                temp_score = (new_r[prev] + new_r[next]) / 2
                #new_r[invalid_idx] = np.round_(temp_score, 2)
                new_r[invalid_idx] = temp_score

        assert f'Monotonic {strictly_monotonic(new_r)}'

        return new_r

    def update_t(self, r):
        return np.array([(r[r_idx] + r[r_idx+1]) / 2 for r_idx in range(len(r) - 1)])

    def get_memeber_num(self, r, mos, print_num=False):
        argmin_idx = cdist(mos.reshape(-1, 1), r.reshape(-1, 1)).argmin(1)
        member_num_list = []

        for r_idx in range(len(r)):
            mos_idx = np.where(argmin_idx == r_idx)[0]
            member_num_list.append(len(mos_idx))

            if print_num:
                print(f"Rpt score: {r[r_idx]:.2f}, Num Mem: {len(mos_idx)}")

        return np.array(member_num_list)

def get_batches_v7_mask_single(cfg, f, f_group, scores, num_list=36, im_num=3):

    # Generate batches
    batch_f_list = []

    total_gt = []

    for test_idx in range(num_list):
        sampled_idx = np.concatenate([np.random.choice(np.where(f_group == g)[0], im_num, replace=False) for g in range(min(cfg.batch_size, len(np.unique(f_group))))])

        f_sampled = f[sampled_idx]
        score_sampled = scores[sampled_idx]
        reidx = np.arange(len(sampled_idx))

        test_idx = np.random.choice(reidx, 10, replace=False)
        ref_idx = np.setdiff1d(reidx, test_idx)

        np.random.shuffle(ref_idx)
        np.random.shuffle(test_idx)

        ref_f = f_sampled[ref_idx]
        test_f = f_sampled[test_idx]

        wo_embed_f = torch.concat([ref_f, test_f])
        scores_tmp = score_sampled[ref_idx.tolist() + test_idx.tolist()]

        permutation = np.arange(len(sampled_idx))
        np.random.shuffle(permutation)

        batch_f_list.append(wo_embed_f[permutation])
        total_gt.append(scores_tmp[permutation].tolist())

    batch_f = torch.stack(batch_f_list).transpose(2, 1)
    total_gt = torch.tensor(total_gt).cuda().float()

    return batch_f, total_gt

def get_left_right_idxs(scores, spv_scores):

    spv_scores = torch.tensor(spv_scores).float().cuda()

    batch_idxs = []
    emb_idxs = []
    emb_scores = []

    left_idxs = []
    left_scores = []

    right_idxs = []
    right_scores = []


    for b in range(len(scores)):
        for i, s in enumerate(scores[b]):

            argsort = torch.abs(spv_scores - s).argsort()
            argmin_idx = argsort[0]
            if spv_scores[argmin_idx] >= s:
                pair_idx = argmin_idx - 1

                left_idx = pair_idx
                right_idx = argmin_idx

                left_score = spv_scores[pair_idx]
                right_score = spv_scores[argmin_idx]
            else:
                pair_idx = argmin_idx + 1

                left_idx = argmin_idx
                right_idx = pair_idx

                left_score = spv_scores[argmin_idx]
                right_score = spv_scores[pair_idx]

            batch_idxs.append(b)

            emb_idxs.append(i)
            emb_scores.append(s)

            left_idxs.append(left_idx)
            left_scores.append(left_score)

            right_idxs.append(right_idx)
            right_scores.append(right_score)

    batch_idxs = torch.tensor(batch_idxs).cuda()
    emb_idxs = torch.tensor(emb_idxs).cuda()
    emb_scores = torch.stack(emb_scores)
    left_idxs = torch.stack(left_idxs)
    left_scores = torch.stack(left_scores)
    right_idxs = torch.stack(right_idxs)
    right_scores = torch.stack(right_scores)

    return batch_idxs, emb_idxs, emb_scores, left_idxs, left_scores, right_idxs, right_scores

def get_pairs_equally_fast_v2(scores):

    B, N = scores.size()

    permutation = np.arange(N)
    np.random.shuffle(permutation)

    batch_idx = np.arange(B).repeat(N).reshape(B, N).flatten()

    base_scores = scores.flatten()
    ref_scores = scores[:, permutation].flatten()

    base_idx = np.tile(np.arange(N), B).reshape(B, N).flatten()
    ref_idx = np.tile(np.arange(N), B).reshape(B, N)[:, permutation].flatten()

    return batch_idx, base_idx, ref_idx, None, tensor2np(base_scores), tensor2np(ref_scores)