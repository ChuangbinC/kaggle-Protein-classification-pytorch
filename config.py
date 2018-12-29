class DefaultConfigs(object):
    base_dir = './data'
    submits = './submits'
    logs = './logs/train'
    checkpoints = './checkpoints'
    test_checkpoint_name = 'm-2018-12-29T04-29-50-0.0716.pth.tar'
    epoch = 50
    lr = 0.03
    weight_decay = 1e-4
    batch_size = 32
    class_num = 28
    valid_size = 0.13
mconfig = DefaultConfigs()
