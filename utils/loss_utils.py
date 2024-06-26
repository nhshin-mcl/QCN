import numpy as np
from einops import rearrange

import torch
import torch.nn as nn

def compute_center_loss_v2(embs, spvs, batch_idxs, emb_idxs, emb_scores, left_idxs, left_scores, right_idxs, right_scores):

    emb_scores_torch = torch.tensor(emb_scores).cuda().float()
    left_scores_torch = torch.tensor(left_scores).cuda().float()
    right_scores_torch = torch.tensor(right_scores).cuda().float()

    right_weight = torch.abs(left_scores_torch - emb_scores_torch)
    left_weight = torch.abs(right_scores_torch - emb_scores_torch)
    l = torch.abs(left_scores_torch - right_scores_torch)

    r_bar = ((spvs[batch_idxs, left_idxs] * left_weight.unsqueeze(-1)) + (spvs[batch_idxs, right_idxs] * right_weight.unsqueeze(-1))) / (l + 1e-10).unsqueeze(-1)
    loss = nn.PairwiseDistance(p=2)(embs[batch_idxs, emb_idxs], r_bar).mean()

    return loss

def compute_mae_loss_v3(embs, spvs, emb_scores, spv_scores):

    dists = torch.cdist(embs, spvs)

    dists_rpt2rpt = torch.cdist(spvs, spvs)
    dists_rpt2rpt = dists_rpt2rpt ** 2


    def get_pos_neg_idxs(dists, scores, spv_scores):

        spv_scores = torch.tensor(spv_scores).float().cuda()

        batch_idxs = []
        emb_idxs = []
        emb_scores = []

        left_idxs = []
        left_scores = []

        right_idxs = []
        right_scores = []

        for b in range(len(scores)):

            dist_pn = torch.stack([dists[b][:, tmp] + dists[b][:, tmp + 1] for tmp in range(len(spv_scores) - 1)]).transpose(1, 0)
            argmin_idx = dist_pn.argmin(1)

            left_idx = argmin_idx
            right_idx = argmin_idx + 1

            left_score = spv_scores[left_idx]
            right_score = spv_scores[right_idx]

            batch_idxs.append([b] * len(argmin_idx))

            emb_idxs.append(torch.arange(len(argmin_idx)))
            emb_scores.append(scores[b])

            left_idxs.append(left_idx)
            left_scores.append(left_score)

            right_idxs.append(right_idx)
            right_scores.append(right_score)

        batch_idxs = torch.tensor(batch_idxs).cuda().flatten()
        emb_idxs = torch.cat(emb_idxs)
        emb_scores = torch.cat(emb_scores)
        left_idxs = torch.cat(left_idxs)
        left_scores = torch.cat(left_scores)
        right_idxs = torch.cat(right_idxs)
        right_scores = torch.cat(right_scores)

        return batch_idxs, emb_idxs, emb_scores, left_idxs, left_scores, right_idxs, right_scores

    batch_idxs, emb_idxs, emb_scores, left_idxs, left_scores, right_idxs, right_scores = get_pos_neg_idxs(dists, emb_scores, spv_scores)

    alpha = ((embs[batch_idxs, emb_idxs] - spvs[batch_idxs, left_idxs]) * (spvs[batch_idxs, right_idxs] - spvs[batch_idxs, left_idxs])).sum(1) / dists_rpt2rpt[batch_idxs, right_idxs, left_idxs]

    left_scores_torch = torch.tensor(left_scores).cuda().float()
    right_scores_torch = torch.tensor(right_scores).cuda().float()
    emb_scores_torch = torch.tensor(emb_scores).cuda().float()

    preds = left_scores_torch + alpha * (right_scores_torch - left_scores_torch)
    loss = nn.SmoothL1Loss()(preds, emb_scores_torch)


    return loss, preds

def compute_rpt_direction_v1(spvs, spv_scores):
    N, L, C = spvs.size()

    def get_forward_backward_idxs(spv_score, batch_size):

        batch_idxs = []
        forward_idxs = []
        backward_idxs = []

        for i in range(len(spv_score)):

            if i == 0:
                forward_idxs.append([i + 1, i])
                backward_idxs.append([i, i + 2])


            elif i == (len(spv_score) - 1):

                forward_idxs.append([i, i - 2])
                backward_idxs.append([i - 1, i])

            else:

                forward_idxs.append([i + 1, i])
                backward_idxs.append([i - 1, i])

        forward_idxs = np.array(forward_idxs)
        backward_idxs = np.array(backward_idxs)

        batch_idxs = np.arange(batch_size).repeat(len(spv_score))
        forward_idxs = np.tile(forward_idxs, (batch_size, 1))
        backward_idxs = np.tile(backward_idxs, (batch_size, 1))

        return batch_idxs, forward_idxs, backward_idxs

    batch_idxs, forward_idxs, backward_idxs = get_forward_backward_idxs(spv_scores, N)

    direction_matrix = spvs.repeat(1, L, 1) - torch.repeat_interleave(spvs, L, dim=1) # ex) idxs (1, 2, 3, 4, 5) - (1, 1, 1, 1, 1)
    direction_matrix = rearrange(direction_matrix, 'N (L l) C -> N L l C', N=N, L=L, C=C)
    direction_matrix = nn.functional.normalize(direction_matrix, dim=-1)

    v_forward = direction_matrix[batch_idxs, forward_idxs[:, 0], forward_idxs[:, 1]]
    v_backward = direction_matrix[batch_idxs, backward_idxs[:, 0], backward_idxs[:, 1]]

    loss = nn.CosineSimilarity(dim=-1)(v_forward, v_backward) + 1

    return loss.sum() / (N * L)

def compute_metric_loss_v1(embs, batch_idx, base_idx, ref_idx, base_scores, ref_scores, spvs, spv_scores, margin, cfg):

    N, L, C = embs.size()
    dists = torch.cdist(spvs,embs)

    def get_pos_neg_idxs(batch_idx, base_idx, ref_idx, base_scores, ref_scores, spv_scores, cfg):
        batch_size = len(base_idx)
        new_batch_idxs = []
        row_idxs = []
        pos_idxs = []
        neg_idxs = []

        sim_new_batch_idxs = []
        sim_row_idxs = []
        sim_pos_idxs = []
        sim_neg_idxs = []

        for i in range(batch_size):
            if base_scores[i] > (ref_scores[i] + cfg.tau):
                fdc_1_idx = np.where(spv_scores <= ref_scores[i])[0]
                fdc_2_idx = np.where(spv_scores >= base_scores[i])[0]

                row_idxs.append(fdc_1_idx)
                pos_idxs.append([ref_idx[i]]*len(fdc_1_idx))
                neg_idxs.append([base_idx[i]]*len(fdc_1_idx))
                new_batch_idxs.append([batch_idx[i]]*len(fdc_1_idx))

                row_idxs.append(fdc_2_idx)
                pos_idxs.append([base_idx[i]]*len(fdc_2_idx))
                neg_idxs.append([ref_idx[i]]*len(fdc_2_idx))
                new_batch_idxs.append([batch_idx[i]] * len(fdc_2_idx))

            elif base_scores[i] < (ref_scores[i] - cfg.tau):
                fdc_1_idx = np.where(spv_scores <= base_scores[i])[0]
                fdc_2_idx = np.where(spv_scores >= ref_scores[i])[0]

                row_idxs.append(fdc_1_idx)
                pos_idxs.append([base_idx[i]] * len(fdc_1_idx))
                neg_idxs.append([ref_idx[i]] * len(fdc_1_idx))
                new_batch_idxs.append([batch_idx[i]] * len(fdc_1_idx))

                row_idxs.append(fdc_2_idx)
                pos_idxs.append([ref_idx[i]] * len(fdc_2_idx))
                neg_idxs.append([base_idx[i]] * len(fdc_2_idx))
                new_batch_idxs.append([batch_idx[i]] * len(fdc_2_idx))

            else:
                sim_row_idxs.append(np.arange(cfg.reference_point_num))
                sim_pos_idxs.append([base_idx[i]]*cfg.reference_point_num)
                sim_neg_idxs.append([ref_idx[i]]*cfg.reference_point_num)
                sim_new_batch_idxs.append([batch_idx[i]] * cfg.reference_point_num)

        row_idxs = np.concatenate(row_idxs)
        pos_idxs = np.concatenate(pos_idxs)
        neg_idxs = np.concatenate(neg_idxs)
        new_batch_idxs = np.concatenate(new_batch_idxs)

        if len(sim_row_idxs) > 0:
            sim_row_idxs = np.concatenate(sim_row_idxs)
            sim_pos_idxs = np.concatenate(sim_pos_idxs)
            sim_neg_idxs = np.concatenate(sim_neg_idxs)
            sim_new_batch_idxs = np.concatenate(sim_new_batch_idxs)
        return new_batch_idxs, row_idxs, pos_idxs, neg_idxs, sim_new_batch_idxs, sim_row_idxs, sim_pos_idxs, sim_neg_idxs

    new_batch_idxs, row_idxs, pos_idxs, neg_idxs, sim_new_batch_idxs, sim_row_idxs, sim_pos_idxs, sim_neg_idxs = get_pos_neg_idxs(batch_idx, base_idx, ref_idx, base_scores, ref_scores, spv_scores, cfg)

    violation = dists[new_batch_idxs, row_idxs, pos_idxs] - dists[new_batch_idxs, row_idxs,neg_idxs]
    violation = violation + margin

    loss = nn.functional.relu(violation)
    return torch.sum(loss) / (N * L)
