from .include.detrpose_hgnetv2 import model, criterion, training_params, postprocessor
from .include.dataset import dataset_train, dataset_val, dataset_test, evaluator 

from src.core import LazyCall as L
from src.nn.optimizer import ModelEMA 
from src.misc.get_param_dicts import get_optim_params

from torch import optim

training_params.output_dir =  "output/detrpose_hgnetv2_n"
training_params.epochs = 160 # 96 + 4 
training_params.use_ema = True
training_params.grad_accum_steps = 1

ema = L(ModelEMA)(
    decay=0.9999,
    warmups=2000
    )

# optimizer params
optimizer = L(optim.AdamW)(
    params=L(get_optim_params)(
        cfg=[
                {
                'params': '^(?=.*backbone).*$',
                'lr': 0.0001
                },
            ],
        # model=model
        ),
    lr=0.0001,
    betas=[0.9, 0.999],
    weight_decay=0.0001
    )

lr_scheduler = L(optim.lr_scheduler.MultiStepLR)(
    # optimizer=optimizer,
    milestones=[1000],
    gamma=0.1
    )

model.backbone.name = 'B0'
model.backbone.use_lab = True
model.backbone.return_idx = [2, 3]
model.encoder.in_channels = [512, 1024]
model.encoder.feat_strides = [16, 32]
model.encoder.n_levels = 2
model.encoder.use_encoder_idx = [1]
model.encoder.depth_mult = 0.5
model.encoder.expansion = 0.34
model.encoder.hidden_dim = 128
model.encoder.dim_feedforward = 512
model.transformer.num_decoder_layers = 3
model.transformer.num_feature_levels = 2
model.transformer.dim_feedforward = 512
model.transformer.feat_strides = [16, 32]
model.transformer.hidden_dim = 128
model.transformer.dec_n_points= 6
model.transformer.use_kan = True
model.transformer.kan_grid = 3

dataset_train.dataset.transforms.policy = {
    'name': 'stop_epoch',
    'ops': ['Mosaic', 'RandomCrop', 'RandomZoomOut'],
    'epoch': [5, 83, 156] # 156 / 2 + 5 = 83
    }
dataset_train.collate_fn.base_size_repeat = None
dataset_train.collate_fn.stop_epoch = 156
