from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "levit_256"
config.resume = False
config.save_all_states = True
config.output = "/home/ubuntu/insightface/recognition/arcface_torch/work_dirs/glint360k_levit_256"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.025
# config.weight_decay= 5e-4
config.batch_size = 256
config.optimizer = "adamw"
config.lr = 0.001
# config.lr = 0.1
config.verbose = 10000
config.dali = False

config.rec = "/media/traindata_ro/users/yl3008/face_recognition/glint360k/glint360k_mask"
# config.rec = "/media/traindata_ro/users/yl3008/face_recognition/glint360k/glint360k"
config.num_classes = 360232
config.num_image = 17091657
config.num_epoch = 40
# config.num_epoch = 24
config.warmup_epoch = 4
# config.warmup_epoch = 0
# config.warmup_epoch = 0.1
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

# config.num_workers = 8