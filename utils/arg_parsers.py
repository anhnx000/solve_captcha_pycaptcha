from argparse import ArgumentParser

def train_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default='exp1')
    parser.add_argument("--gpus", default=0)
    parser.add_argument("--log_dir", default='./logs/')
    parser.add_argument("--save_path", default='./checkpoint/')
    parser.add_argument("--model_name", default='resnet18') # resnet18, resnet34, EfficientNetV2-S, vit, mobilenet
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint file to resume training from")
    args = parser.parse_args()
    return args

def test_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--test_name", default='test')
    parser.add_argument("--log_dir", default='./logs/')
    parser.add_argument("--ckpt", default='./logs/main/version_2/checkpoints/epoch=4-step=1954.ckpt')
    args = parser.parse_args()
    return args

def predict_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--input", default='./input')
    parser.add_argument("--ckpt", default='./logs/main/version_2/checkpoints/epoch=4-step=1954.ckpt')
    args = parser.parse_args()
    return args
