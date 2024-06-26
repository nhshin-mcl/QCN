import pandas as pd

def get_df_v1(cfg, is_train=False):

    if cfg.dataset_name == 'PaQ2PiQ':

        if is_train:
            train_image_path = cfg.dataset_root + 'Images/'
            train_df_path = cfg.datasplit_root + f'train_image_split.xlsx'

        else:
            ref_image_path = cfg.dataset_root + 'Images/'
            ref_df_path = cfg.datasplit_root + f'train_image_split.xlsx'

            test_image_path = cfg.dataset_root + 'Images/'
            test_df_path = cfg.datasplit_root + f'{cfg.training_scheme}_image_split.xlsx'

    elif cfg.dataset_name == 'SPAQ':

        if is_train:
            train_image_path = cfg.dataset_root + 'Images_resized_384/'
            train_df_path = cfg.datasplit_root + 'SPAQ_train_split_%d.xlsx'%(cfg.split)

        else:
            ref_image_path = cfg.dataset_root + 'Images_resized_384/'
            ref_df_path = cfg.datasplit_root + 'SPAQ_train_split_%d.xlsx'%(cfg.split)

            test_image_path = cfg.dataset_root + 'Images_resized_384/'
            test_df_path = cfg.datasplit_root + 'SPAQ_test_split_%d.xlsx'%(cfg.split)

    elif cfg.dataset_name == 'KonIQ10K':

        if is_train:
            train_image_path = cfg.dataset_root + '512x384/'
            if cfg.training_scheme == 'koniq10k':
                train_df_path = cfg.datasplit_root + 'koniq10k_distributions_sets.csv'

            elif cfg.training_scheme == 'random_split':
                train_df_path = cfg.datasplit_root + '/KonIQ10K_train_split_%d.csv' % (cfg.split)

        else:
            ref_image_path = cfg.dataset_root + '512x384/'
            if cfg.training_scheme == 'koniq10k':
                ref_df_path = cfg.datasplit_root + 'koniq10k_distributions_sets.csv'

            elif cfg.training_scheme == 'random_split':
                ref_df_path = cfg.datasplit_root + f'/KonIQ10K_train_split_%d.csv' % (cfg.split)

            test_image_path = cfg.dataset_root + '512x384/'
            if cfg.training_scheme == 'koniq10k':
                test_df_path = cfg.datasplit_root + 'koniq10k_distributions_sets.csv'

            elif cfg.training_scheme == 'random_split':
                test_df_path = cfg.datasplit_root + '/KonIQ10K_test_split_%d.csv' % (cfg.split)


    elif cfg.dataset_name == 'CLIVE':

        if is_train:
            train_image_path = cfg.dataset_root + 'Images/'
            train_df_path = cfg.datasplit_root + 'CLIVE_training_split_%d.xlsx'%(cfg.split)

        else:
            ref_image_path = cfg.dataset_root + 'Images/'
            ref_df_path = cfg.datasplit_root + 'CLIVE_training_split_%d.xlsx'%(cfg.split)

            test_image_path = cfg.dataset_root + 'Images/'
            test_df_path = cfg.datasplit_root + 'CLIVE_test_split_%d.xlsx'%(cfg.split)

    elif cfg.dataset_name == 'BID':

        if is_train:
            train_image_path = cfg.dataset_root + 'Images_resized/'
            train_df_path = cfg.datasplit_root + 'BID_train_split_%d.xlsx'%(cfg.split)

        else:
            ref_image_path = cfg.dataset_root + 'Images_resized/'
            ref_df_path = cfg.datasplit_root + 'BID_train_split_%d.xlsx'%(cfg.split)

            test_image_path = cfg.dataset_root + 'Images_resized/'
            test_df_path = cfg.datasplit_root + 'BID_test_split_%d.xlsx'%(cfg.split)

    else:
        raise ValueError(f'Undefined database ({cfg.dataset_name}) has been given')


    if is_train:
        try:
            return train_image_path, pd.read_excel(train_df_path)
        except:
            return train_image_path, pd.read_csv(train_df_path)
    elif is_train is False:
        try:
            return ref_image_path, pd.read_excel(ref_df_path), test_image_path, pd.read_excel(test_df_path)
        except:
            return ref_image_path, pd.read_csv(ref_df_path), test_image_path, pd.read_csv(test_df_path)
    else:
        raise ValueError(f'Undefined mode ({is_train}) has been given')

