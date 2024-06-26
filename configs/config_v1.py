import os
from utils.util import get_current_time, write_log

class ConfigV1:
    def __init__(self):
        # dataset
        self.dataset_name = 'SPAQ'
        self.dataset_root = f'/home/nhshin/dataset/IQA/{self.dataset_name}/'
        self.datasplit_root = f'./datasplit/{self.dataset_name}/'
        self.training_scheme = 'random_split' # for KonIQ10K dataset
        self.split = 1

        # network
        self.backbone = 'resnet50'
        self.model_name = 'CTV29'

        # Score pivots
        self.spv_num = 101

        # exp description
        self.exp_name = ''

        # encoder
        self.image_size = 384
        self.reduced_dim = 256

        # comparison transformer
        self.dec_layers = [1] * 3

        # train & test
        self.batch_size = 18
        self.test_batch_size = 10
        self.num_list = 9
        self.start_iter = 0
        self.start_eval = 60
        self.eval_freq = 5

        self.im_list_len = 150
        self.im_num = 2

        # hyperparameters
        self.tau = 0
        self.margin = 0.1

        # optimization
        self.optim = 'AdamW'
        self.scheduler = 'cosinewarmup'
        self.lr = 1e-5 * 5
        self.weight_decay = 1e-4 * 5

        self.epoch = 100
        self.test_first = True

        # misc
        self.num_workers = 0
        self.gpu = '1'
        self.wandb = False

        # logging
        self.save_folder_parent = './results/'
        self.save_folder = f'{self.save_folder_parent}/{self.dataset_name}/{self.model_name}/Back_{self.backbone}_M_{self.model_name}_C{self.reduced_dim}_T{self.tau:.2f}_{self.exp_name}_{get_current_time()}'
        os.makedirs(self.save_folder, exist_ok=True)

        self.load = False
        if self.load:
            self.ckpt_file = 'SRCC_Epoch_94_SRCC_0.9246_PCC_0.9265_MAE_6.1870.pth'
            self.init_model = f'./ckpt/{self.dataset_name}/Split_{self.split}/{self.ckpt_file}'
        #self.log_file = self.log_configs()

    def log_configs(self, log_file='log.txt'):
        if os.path.exists(f'{self.save_folder}/{log_file}'):
            log_file = open(f'{self.save_folder}/{log_file}', 'a')
        else:
            log_file = open(f'{self.save_folder}/{log_file}', 'w')

        write_log(log_file, '------------ Options -------------')
        for k in vars(self):
            write_log(log_file, f'{str(k)}: {str(vars(self)[k])}')
        write_log(log_file, '-------------- End ----------------')

        # log_file.close()
        return log_file


if __name__ == "__main__":
    c = ConfigV1()
    print('debug... ')


