import os
import sys
import math

from scipy.stats import spearmanr, pearsonr
import numpy as np
import wandb
from einops import rearrange

import torch
from torch.utils.data import DataLoader
import torchvision

from network.network_utils import build_model
from network.optimizer_utils import get_optimizer, get_scheduler
from dataloaders import dataloader_gen

from utils.loss_utils import compute_center_loss_v2, compute_rpt_direction_v1, compute_mae_loss_v3, compute_metric_loss_v1
from utils.train_utils import get_batches_v7_mask_single, Maxlloyd, get_left_right_idxs, get_pairs_equally_fast_v2

from utils.test_utils import compute_score_v1

from utils.util import load_model, save_model
from utils.util import set_wandb, tensor2np, write_log


def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    train_dataset, test_ref_loader, test_loader = dataloader_gen.gen_dataloader(cfg)

    train_dataset_moses = train_dataset.df['MOS'].values
    cfg.n_scores = len(np.unique(train_dataset_moses))

    # pivot score setting
    maxlloyd = Maxlloyd(train_dataset_moses, rpt_num=cfg.spv_num)
    cfg.score_pivot_score = maxlloyd.get_new_rpt_scores()
    cfg.reference_point_num = len(cfg.score_pivot_score)

    cfg.log_file = cfg.log_configs()

    write_log(cfg.log_file, f'[*] {cfg.n_scores} scores exist.')
    write_log(cfg.log_file, f'[*] {cfg.reference_point_num} reference points.')

    net = build_model(cfg)

    if cfg.wandb:
        set_wandb(cfg)
        wandb.watch(net)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        net = net.cuda()

    optimizer = get_optimizer(cfg, net)
    lr_scheduler = get_scheduler(cfg, optimizer)

    if cfg.load:
        load_model(cfg, net, optimizer=optimizer, load_optim_params=False)

    if cfg.test_first:
        net.eval()
        srcc, pcc, mae = evaluation(cfg, net, test_ref_loader, test_loader)
        sys.stdout.write(f'[SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')

    best_srcc = 0.85
    best_pcc = 0.85

    best_srcc_total_results = []
    best_pcc_total_results = []

    best_srcc_epoch = -1
    best_pcc_epoch = -1

    log_dict = dict()
    for epoch in range(0, cfg.epoch):
        net.train()
        net.encoder.eval()

        if (epoch + 1) <= 5:
            uniform_select = True
        else:
            uniform_select = False

        train_dataset.get_pair_lists(batch_size=cfg.batch_size, batch_list_len=cfg.im_list_len, im_num=cfg.im_num, uniform_select=uniform_select)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=cfg.num_workers, shuffle=False, pin_memory=True, drop_last=True)

        mae_loss, center_loss, order_loss, metric_loss = train(cfg, net, optimizer, train_loader, epoch)
        write_log(cfg.log_file, f'\nEpoch: {(epoch + 1):d} MAE Loss: {mae_loss:.3f}, Center Loss: {center_loss:.3f}, Order Loss: {order_loss:.3f}, Metric Loss: {metric_loss:.3f}\n')

        if cfg.wandb:
            log_dict['Epoch'] = epoch
            log_dict['LR'] = lr_scheduler.get_lr()[0] if lr_scheduler else cfg.lr

        if ((epoch + 1) == 1) | (((epoch + 1) >= cfg.start_eval) & ((epoch + 1) % cfg.eval_freq == 0)):

            net.eval()
            srcc, pcc, mae = evaluation(cfg, net, test_ref_loader, test_loader)
            sys.stdout.write(f'\n[SRCC: {srcc:.4f}] [PCC: {pcc:.4f}] [MAE: {mae:.4f}] \n')

            if srcc > best_srcc:
                best_srcc = srcc
                best_srcc_epoch = epoch
                best_srcc_total_results = [srcc, pcc, mae]
                save_model(cfg, net, optimizer, epoch, [srcc, pcc, mae], criterion='SRCC')

            if pcc > best_pcc:
                best_pcc = pcc
                best_pcc_epoch = epoch
                best_pcc_total_results = [srcc, pcc, mae]
                save_model(cfg, net, optimizer, epoch, [srcc, pcc, mae], criterion='PCC')

            if cfg.wandb:
                log_dict['Test/SRCC'] = srcc
                log_dict['Test/PCC'] = pcc
                log_dict['Test/MAE'] = mae

        if cfg.wandb:
            wandb.log(log_dict)

        print('lr: %.6f' % (optimizer.param_groups[0]['lr']))
        if lr_scheduler:
            lr_scheduler.step()

    write_log(cfg.log_file, 'Training End')
    write_log(cfg.log_file,
              'Best SRCC / Epoch: %d\tSRCC: %.4f\tPCC: %.4f\tMAE: %.4f' % (best_srcc_epoch, best_srcc_total_results[0], best_srcc_total_results[1], best_srcc_total_results[2]))
    write_log(cfg.log_file,
              'Best PCC / Epoch: %d\tSRCC: %.4f\tPCC: %.4f\tMAE: %.4f' % (best_pcc_epoch, best_pcc_total_results[0], best_pcc_total_results[1], best_pcc_total_results[2]))
    print(cfg.save_folder)

def train(cfg, net, optimizer, data_loader, epoch):
    mae_losses = 0
    order_losses = 0
    center_losses = 0
    metric_losses = 0

    dataloader_iterator = iter(data_loader)

    for idx in range(len(dataloader_iterator) // cfg.batch_size):

        optimizer.zero_grad()

        f = []
        scores = []
        groups = []
        for dl_iter in range(cfg.batch_size):
            sample = next(dataloader_iterator)

            scores_tmp = torch.cat([sample[f'img_{im_idx}_mos'] for im_idx in range(cfg.im_num)])
            scores_tmp = scores_tmp.cuda().float()
            scores.append(scores_tmp)

            groups_tmp = torch.cat([sample[f'img_{im_idx}_group'] for im_idx in range(cfg.im_num)])
            groups_tmp = groups_tmp.cuda()
            groups.append(groups_tmp)

            # Extract features
            for im_idx in range(cfg.im_num):
                img = sample[f'img_{im_idx}'].cuda()
                img_f = net('extraction', {'img': img}).squeeze()
                f.append(img_f)

        f = torch.stack(f)

        scores = torch.cat(scores)
        groups = torch.cat(groups)

        # Generate batches
        f_new, total_gt = get_batches_v7_mask_single(cfg, f, tensor2np(groups), scores, num_list=cfg.num_list, im_num=cfg.im_num)
        batch_idxs, emb_idxs, emb_scores, left_idxs, left_scores, right_idxs, right_scores = get_left_right_idxs(scores=total_gt, spv_scores=cfg.score_pivot_score)

        # Obtain updated features and score pivots
        f_new, score_pivots = net('get_cluster', {'f': f_new})

        # Computes losses
        mae_loss, preds = compute_mae_loss_v3(embs=rearrange(f_new, 'b c l -> b l c'),
                                              spvs=rearrange(score_pivots, 'b c l -> b l c'),
                                              emb_scores=total_gt,
                                              spv_scores=cfg.score_pivot_score,
                                              )


        center_loss = compute_center_loss_v2(embs=rearrange(f_new, 'b c l -> b l c'),
                                             spvs=rearrange(score_pivots, 'b c l -> b l c'),
                                             batch_idxs=batch_idxs,
                                             emb_idxs=emb_idxs,
                                             emb_scores=emb_scores,
                                             left_idxs=left_idxs,
                                             left_scores=left_scores,
                                             right_idxs=right_idxs,
                                             right_scores=right_scores,
                                             )

        order_loss = compute_rpt_direction_v1(spvs=rearrange(score_pivots, 'b c l -> b l c'),
                                              spv_scores=cfg.score_pivot_score,
                                              )

        batch_idx, base_idx, ref_idx, _, base_scores, ref_scores = get_pairs_equally_fast_v2(scores=total_gt)

        metric_loss = compute_metric_loss_v1(embs=rearrange(f_new, 'b c l -> b l c'),
                                             batch_idx=batch_idx,
                                             base_idx=base_idx,
                                             ref_idx=ref_idx,
                                             base_scores=base_scores,
                                             ref_scores=ref_scores,
                                             spvs=rearrange(score_pivots, 'b c l -> b l c'),
                                             spv_scores=cfg.score_pivot_score,
                                             margin=cfg.margin,
                                             cfg=cfg)

        if (epoch + 1) >= 10:
            loss = mae_loss + center_loss + order_loss + metric_loss
        else:
            loss = center_loss + order_loss + metric_loss

        loss.backward()
        optimizer.step()

        tot_mae_loss = (mae_loss.item())
        tot_center_loss = (center_loss.item())
        tot_order_loss = (order_loss.item())
        tot_metric_loss = (metric_loss.item())

        mae_losses += tot_mae_loss
        center_losses += tot_center_loss
        order_losses += tot_order_loss
        metric_losses += tot_metric_loss

        sys.stdout.write(
            f'\r[Epoch {epoch + 1}/{cfg.epoch}] [Batch {idx + 1}/{cfg.im_list_len}] [Loss {loss.item():.2f}] [MAE {tot_mae_loss:.2f}] [Center {tot_center_loss:.2f}] [Order {tot_order_loss:.2f}] [Metric {tot_metric_loss:.2f}]')

    return mae_losses / (idx + 1), center_losses / (idx + 1), order_losses / (idx + 1), metric_losses / (idx + 1)

def evaluation(cfg, net, ref_data_loader, data_loader):
    net.eval()
    test_mos_gt = data_loader.dataset.df_test['MOS'].values

    preds_list = []
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            # Extract features of auxiliary images
            f_list = []
            for idx, sample in enumerate(ref_data_loader):
                if idx % 1 == 0:
                    sys.stdout.write(f'\rExtract Aux Img Features... [{idx + 1}/{len(ref_data_loader)}]')

                image = sample[f'img'].cuda()
                f = net('extraction', {'img': image})
                f_list.append(f)

            aux_f = torch.cat(f_list)
            aux_f = aux_f.squeeze()
            aux_f = aux_f.transpose(1, 0)

            # Extract features of test images
            test_f_list = []
            for idx, sample in enumerate(data_loader):

                if idx % 1 == 0:
                    sys.stdout.write(f'\rExtract Test Img Features... [{idx + 1}/{len(data_loader)}]')

                image = sample[f'img'].cuda()
                f = net('extraction', {'img': image})
                f_hflip = net('extraction', {'img': torchvision.transforms.functional.hflip(image)})

                test_f_list.append(f)
                test_f_list.append(f_hflip)

            test_f = torch.cat(test_f_list)
            test_f = test_f.squeeze()
            test_f = rearrange(test_f, '(N Cr) C -> N Cr C', N=len(test_mos_gt), C=cfg.reduced_dim).mean(1)
            test_f = test_f.transpose(1, 0)

            # Set # of iterations
            n_iter = int(math.ceil(len(test_mos_gt) / cfg.test_batch_size))
            crop_num = 1
            start = 0

            for idx in range(n_iter):
                if idx % 1 == 0:
                    sys.stdout.write(f'\rTesting... [{idx + 1}/{n_iter}]')

                batch = min(cfg.test_batch_size, len(test_mos_gt) - len(preds_list))

                f = torch.cat([aux_f.unsqueeze(0).repeat(batch, 1, 1),
                               rearrange(test_f[:, start:start + (batch * crop_num)], 'C (N L) -> N C L', N=batch, L=crop_num)], dim=-1)

                # Obtain updated features and score pivots
                f, score_pivots = net('get_cluster', {'f': f})

                # Estimate quality scores
                preds = compute_score_v1(embs=rearrange(f, 'b c l -> b l c')[:, -1:],
                                         spv=rearrange(score_pivots, 'b c l -> b l c'),
                                         emb_scores=torch.tensor(test_mos_gt[start:(start + batch)].reshape(-1, 1)).cuda().float(),
                                         spv_scores=cfg.score_pivot_score,
                                         )

                preds_list.extend(preds.tolist())
                start += (batch * crop_num)

    preds_np = np.array(preds_list)

    srcc = spearmanr(preds_np, test_mos_gt)[0]
    pcc = pearsonr(preds_np, test_mos_gt)[0]
    mae = np.abs(preds_np - test_mos_gt).mean()

    write_log(cfg.log_file, f'\nTest MAE: {mae: .4f} SRCC: {srcc: .4f} PCC: {pcc: .4f}')

    return srcc, pcc, mae



if __name__ == "__main__":
    from configs.config_v1 import ConfigV1 as Config

    cfg = Config()
    main(cfg)