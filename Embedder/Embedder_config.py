import torch
from types import SimpleNamespace


config = SimpleNamespace()

config.EMB_INIT = False
config.USE_ARCFACE = True
#
# config.INIT_nnEmb = False
# config.INIT_RELATIVE = True
#

# GPU / WORKERS / BATCH
config.GPUS = '1'
config.WORKERS = 0
config.DEVICE = torch.device(f"cuda:{config.GPUS}" if torch.cuda.is_available() else "cpu")
config.BATCH_SIZE = 32

# MODE
config.TASK_MODE = 'VAL' # ['TRAIN', 'VAL']
config.EMB_MODE = 'RELATIVE_BASIS' # ['RELATIVE_BASIS', 'RELATIVE']
config.BASIS_FREEZE = False     # False -> False /
config.RELATIVE_FREEZE = True   # False -> True

# SEED
config.SEED = 478

# DATA
# AT SERVER 1
config.T_DATA_PATH = '/storage/jysuh/BERTSUMFORHPE/embedder_dataset/train.pkl'
config.T_VOCAB_PATH = '/storage/jysuh/BERTSUMFORHPE/embedder_dataset/train_vocab.pkl'
# config.V_DATA_PATH = '/storage/jysuh/BERTSUMFORHPE/embedder_dataset/valid.pkl'
config.V_DATA_PATH = '/home/jysuh/PycharmProjects/ex_cls_test/BERT_Demo_dataset.pkl'
config.V_VOCAB_PATH = '/storage/jysuh/BERTSUMFORHPE/embedder_dataset/valid_vocab.pkl'

#
config.IMG_SIZE = [1920, 1080]
config.NUM_JOINTS = 22  # 0, 1 = PAD, SEP, others joints
config.MAX_FRAMES = 21
config.CLASS_NUM = 41 if config.TASK_MODE == 'TRAIN' else 27
config.JOINTS_NAME = [
    'Left Shoulder', 'Right Shoulder',
    'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist',
    'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee',
    'Left Ankle', 'Right Ankle',
    'Neck', 'Left Palm',
    'Right Palm', 'Back',
    'Waist', 'Left Foot',
    'Right Foot', 'Head'
    ]

# PRETRAINED MODEL PATH
config.ARCFACE_PARAM = {}
if config.EMB_MODE == 'RELATIVE_BASIS':
    config.USE_EMBEDDING = True
    # RELATIVE PATH
    config.PRETRAINED_PATH = '/storage/jysuh/BERTSUMFORHPE/checkpoint/jy_weight/[Basis+Relative] LAYERS_NUM:4 DIM:768 ACT:GELU s:10 m:0.1/metric_learning_model.pth.tar'
    # nn.EMBEDDING PATH
    config.PRETRAINED_EMB_PATH = '/storage/jysuh/BERTSUMFORHPE/checkpoint/jy_weight/[Basis+Relative] LAYERS_NUM:4 DIM:768 ACT:GELU s:10 m:0.1/nn_embedding_model.pt'
    config.ARCFACE_PARAM['m'] = 0.1
    config.ARCFACE_PARAM['s'] = 10

elif config.EMB_MODE == 'RELATIVE':
    config.USE_EMBEDDING = False
    ## CHANGE PATH) LAYER_NUM = {2,4,6}
    config.PRETRAINED_PATH = '/storage/jysuh/BERTSUMFORHPE/checkpoint/jy_weight/[Relative] LAYERS_NUM:6 DIM:768 ACT:GELU s:10 m:0.1/metric_learning_model.pth.tar'

    # config.ARCFACE_PARAM['m'] = 0.45
    # config.ARCFACE_PARAM['s'] = 50
    config.ARCFACE_PARAM['m'] = 0.1
    config.ARCFACE_PARAM['s'] = 10

# ========================
OUT_FEAT = int(config.PRETRAINED_PATH.split('/')[6].split()[2].split(':')[-1])
NUM_LAYER = int(config.PRETRAINED_PATH.split('/')[6].split()[1].split(':')[-1])
ACTIV = config.PRETRAINED_PATH.split('/')[6].split()[3].split(':')[-1]
# ====== [HR Debug] ========
# OUT_FEAT = int(config.PRETRAINED_PATH.split('/')[6].split()[3].split(':')[1])
# NUM_LAYER = int(config.PRETRAINED_PATH.split('/')[6].split()[2].split(':')[1])
# ACTIV = config.PRETRAINED_PATH.split('/')[6].split()[4].split(':')[-1]
# ==========================
config.IN_FEAT = 4
config.OUT_FEAT = OUT_FEAT
config.NUM_LAYER = NUM_LAYER # EMB_LAYER
config.ACTIV = ACTIV
#

# SAVE DIR NAME
if config.EMB_MODE == 'RELATIVE_BASIS':
    config.DIR_PATH = config.EMB_MODE + "/basis_{}".format(config.BASIS_FREEZE) \
                                      + "/relative_{}".format(config.RELATIVE_FREEZE) \
                                      + "NUM_EMB_LAYER:{}".format(NUM_LAYER)\


elif config.EMB_MODE == 'RELATIVE':
    config.DIR_PATH = config.EMB_MODE + "/relative_{}".format(config.RELATIVE_FREEZE)\
                                      + "NUM_LAYER:{}".format(NUM_LAYER) \


print()
print('####### CONFIG #######')
print('BATCH: {}'.format(config.BATCH_SIZE))
print('BASIS_FREEZE: {}'.format(config.BASIS_FREEZE))
print('RELATIVE_FREEZE: {}'.format(config.RELATIVE_FREEZE))
print('EMB_MODE: {}'.format(config.EMB_MODE))
print('EMB_INIT: {}'.format(config.EMB_INIT))
print('config.USE_ARCFACE: {}'.format(config.USE_ARCFACE))
print('EMB_LAYER: {}'.format(config.NUM_LAYER))
if config.EMB_MODE == 'RELATIVE_BASIS':
    print('BASIS_WEIGHT:{}'.format(config.PRETRAINED_PATH))
    print('RELATIVE_WEIGHT:{}'.format(config.PRETRAINED_EMB_PATH))
elif config.EMB_MODE == 'RELATIVE':
    print('BASIS_WEIGHT_USAGE:{}'.format(config.USE_EMBEDDING))
    print('RELATIVE_WEIGHT:{}'.format(config.PRETRAINED_PATH))

if config.USE_ARCFACE:
    print(config.ARCFACE_PARAM)

