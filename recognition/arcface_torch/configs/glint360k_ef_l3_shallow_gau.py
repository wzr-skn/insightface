from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "ef_l3_shallow_gau"
# config.load_from = "/home/ubuntu/insightface/recognition/arcface_torch/work_dirs/glint360k_ef_l3_shallow_aug_s/checkpoint_gpu_0.pt"
config.load_from = None
config.resume = False
config.save_all_states = True
config.output = "/home/ubuntu/insightface/recognition/arcface_torch/work_dirs/glint360k_ef_l3_shallow_gau_aug_s"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.05
# config.weight_decay= 5e-4
config.batch_size = 256
config.optimizer = "adamw"
config.lr = 0.001
# config.lr = 0.1
config.verbose = 10000
config.dali = False
config.dali_aug = False
config.aug_pipeline = {'MaskAugmentationAdd': {'mask_names': ['mask_white', 'mask_blue', 'mask_black', 'mask_green'],
                                               'mask_probs': [0.4, 0.4, 0.1, 0.1], 'h_low': 0.33,
                                               'h_high': 0.4, 'p': 0.3},
                       # 'CutOut': {'n_holes': 5, 'p': 0.8},
                       # 'GridMask': {'use_h': True, 'use_w': True, 'rotate': 1, 'ratio': 0.15, 'p': 0.2},
                       # 'BlurByBlock': {'blur_limit': (1, 3)},
                       # 'Enlight': {'strength_light': 80, 'p': 0.4},
                       # 'Shadow': {'strength_light': 80, 'p': 0.7}
                       }

config.rec = "/media/traindata_ro/users/yl3008/face_recognition/glint360k/glint360k_mask"
# config.add_rec = "/media/traindata_ro/users/yl4957/face_recognition/glint_part"
config.add_rec = None
# config.rec = "/media/traindata_ro/users/yl3008/face_recognition/glint360k/glint360k"
config.keep_max_yaw_val = 40
config.max_yaw = 70
config.keep_max_pitch_val = 25
config.max_pitch = 50
config.keep_max_yaw_and_pitch_val = [30, 15]
config.max_yaw_and_pitch_val = [60, 40]
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 40
# config.num_epoch = 24
config.warmup_epoch = 4
# config.warmup_epoch = 0
# config.warmup_epoch = 0.1
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

# config.num_workers = 8