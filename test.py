from model.model import captcha_model, model_conv, model_resnet
from data.datamodule import captcha_dm
from utils.arg_parsers import test_arg_parser
import pytorch_lightning as pl
import wandb

def test(args):
    dm = captcha_dm()
    model = captcha_model.load_from_checkpoint(args.ckpt, model=model_resnet())
    wandb_logger = pl.loggers.WandbLogger(
        project="captcha-test",
        name=args.test_name,
        save_dir=args.log_dir,
        version=2
    )
    trainer = pl.Trainer(deterministic=True,
                        #  gpus=-1,
                        #  auto_select_gpus=True,
                         precision=32,
                         logger=wandb_logger,
                         fast_dev_run=False,
                         max_epochs=5,
                         log_every_n_steps=50,
                         stochastic_weight_avg=True
                         )
    trainer.test(model, dm)

if __name__ == "__main__":
    args = test_arg_parser()
    test(args)
