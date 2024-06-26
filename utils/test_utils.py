import torch
import torch.nn as nn

def compute_score_v1(embs, spv, emb_scores, spv_scores):

    dists = torch.cdist(embs, spv)

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

    x = embs[batch_idxs, emb_idxs]

    r_left = spv[batch_idxs, left_idxs]
    r_right = spv[batch_idxs, right_idxs]

    alpha = ((x - r_left) * (r_right - r_left)).sum(1) / (nn.PairwiseDistance(p=2)(r_right, r_left) ** 2)
    preds = left_scores + alpha * (right_scores - left_scores)

    return preds
