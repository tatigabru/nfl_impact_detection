from collections import namedtuple
from . import *
import os


opt_tuple = namedtuple('Opt', ['model', 'n_classes', 'image_size', 'groups', 'width_mult'])


def get_model(model_name, num_classes, clip_image_size, checkpoint_path=None, device='cuda:0'):
    #opt = opt_tuple(model_name, num_classes, clip_image_size[0], 3, 1)
    opt = parse_opts()
    opt.model = model_name
    opt.n_classes = 600 #num_classes
    opt.n_finetune_classes = num_classes
    opt.image_size = clip_image_size[0]
    opt.sample_size = clip_image_size[0]
    opt.sample_duration = 16
    if checkpoint_path is None:
        opt.pretrain_path = get_pretrained_model_path(model_name)
    else:
        opt.pretrain_path = checkpoint_path
    opt.pretrain_path = None
    opt.arch = model_name
    opt.width_mult = 1.5
    opt.model_depth = 50
    #opt.ft_portion = 'last_layer'
    print(opt.pretrain_path)
    if model_name == 'efficientnet3d_b0':
        model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': num_classes, 'image_size': 64},
                                         in_channels=3)
        model.to(device)
        model_parameters = model.parameters()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
    if model_name in ('shufflenet', 'shufflenetv2', 'shufflenet2fc', 'resnet'):
        model, model_parameters = generate_model(opt, device)

    return model, model_parameters


def get_pretrained_model_path(model='shufflenet', dataset='kinetics', width_mult=1.5):
    data_dir = os.environ["KAGGLE_2020_NFL"]
    if model == 'shufflenet':
        model_dir = os.path.join(os.path.join(data_dir, 'pretrained_3d'), 'efficient_3dcnns')
        model_fn = f'_'.join([dataset, model, str(width_mult) + 'x', 'G3',
                                   'RGB', str(16)]) + '_best.pth'
    elif model == 'resnet':
        model_dir = os.path.join(os.path.join(data_dir, 'pretrained_3d'), 'resnets')
        model_fn  = 'r3d50_K_200ep.pth'
    elif model == 'efficientnet3d_b0':
        return ''
    return os.path.join(model_dir, model_fn)