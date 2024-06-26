import os
from datetime import datetime
from collections import OrderedDict
import wandb
import torch


def get_current_time():
    _now = datetime.now()
    _now = str(_now)[:-7]
    return _now

def load_model(args, net, optimizer=None, load_optim_params=False):
    checkpoint = torch.load(args.init_model, map_location=torch.device("cuda:%s" % (0) if torch.cuda.is_available() else "cpu"))
    model_dict = net.state_dict()

    new_model_state_dict = OrderedDict()
    for k, v in model_dict.items():
        if k in checkpoint['model_state_dict'].keys():
            new_model_state_dict[k] = checkpoint['model_state_dict'][k]

        else:
            new_model_state_dict[k] = v
            print(f'Not Loaded\t{k}')
    net.load_state_dict(new_model_state_dict)

    print("=> loaded checkpoint '{}'".format(args.init_model))

    if load_optim_params == True:

        optimizer_dict = optimizer.state_dict()
        optimizer_dict.update(checkpoint['optimizer_state_dict'])
        optimizer.load_state_dict(optimizer_dict)

        print("=> loaded optimizer params '{}'".format(args.init_model))

def save_model(args, net, optimizer, epoch, results, criterion):

    srcc, pcc, mae = results[0], results[1], results[2]

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.save_folder + '/' + '%s_Epoch_%d'%(criterion, epoch) + '_SRCC_%.4f_PCC_%.4f_MAE_%.4f' % (srcc, pcc, mae) + '.pth'))
    print('Saved model to ' + args.save_folder + '/' + '%s_Epoch_%d'%(criterion, epoch) + '_SRCC_%.4f_PCC_%.4f_MAE_%.4f' % (srcc, pcc, mae) + '.pth')


def tensor2np(tensor):
    numpy_data = tensor.cpu().detach().numpy()
    return numpy_data

def set_wandb(cfg):
    wandb.login(key='')
    wandb.init(project=f'{cfg.dataset_name}', tags=[cfg.exp_name])
    wandb.config.update(cfg)
    wandb.save('*.py')
    wandb.run.save()

def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)
