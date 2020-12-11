#
# import logging

import datetime
import dateutil.tz
import os
import hydra
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.core.memory import ModelSummary
# from pytorch_lightning import _logger as log
from pytorch_lightning.profiler import AdvancedProfiler

from utils.algo_selector import algo_selector
from utils.config import Config
from utils.printcallback import PrintingCallback


@hydra.main(config_path='configs/config.yaml')
def main(con: DictConfig):
    if con.name is None:
        con.name = con.alias + "_" + str(con.seed) + "_" + \
                   datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%m-%d-%H-%M-%S-%f-%Z')

    print(con.pretty())
    Config.set_seed(con.seed)
    # TODO add functionality to set this using config. Also consider using checkpoint callback
    # TODO os.environ doesn't seem to work on server, still uploading checkpoints.
    os.environ['WANDB_IGNORE_GLOBS'] = "*.ckpt"
    os.environ['WANDB_IGNORE_GLOBS'] += "*.mp4"
    if con.wandb_logging:
        logger = WandbLogger(name=con.name, project=con.project, tags=con.tags)
        print(logger.experiment.id)  # This print ensures that wandb init is called
        # calling wandb init produces version id which can be used below
        # Creating different path to bypass ckpt upload to wandb to save bandwidth
        path = "CLAD-pytorch/version" + str(con.seed) + "_" + \
               logger.version + "/{epoch}"  # None #logger.experiment.dir
    else:
        logger = True
        path = None
    # wandb.init(config=config)
    # TODO movee this logic elsewehre (like config.freeze)
    # converting gpus to list if using GPUs
    if not type(con.gpus) == int:
        gpus = list(con.gpus)
    else:
        gpus = con.gpus

    model = algo_selector(con)
    # TODO perhaps psasing this class from config is better? Also pass algo from config
    # log.info("test printing")
    # log.warn("test printing: warn")
    # log.error("test printing: error")
    # TODO: figure out how to change log level from info to warn
    # print("Log Level: ",logging.getLogger("pytorch_lighting").setLevel("WARNING"))

    # print("model summary", ModelSummary(model, 'full').__str__())
    # profiler = AdvancedProfiler()

    # TODO update pl version to use new features
    if con.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            filepath=path,
            save_top_k=-1,  # Unused feature
            period=con.checkpoint_period,
            verbose=True,
            monitor='val_loss',  # Not used
            mode='min'  # Not used
        )
    else:
        checkpoint_callback = False

    trainer = pl.Trainer(
        gpus=gpus,
        # TODO: pass these params as dictionary or args. otherwise put default values for backend etc.
        distributed_backend=con.backend,
        max_epochs=con.max_epochs,
        early_stop_callback=con.early_stop_callback,
        val_check_interval=con.epoch_length,  # Checks validation after every epoch
        profiler=con.profiler,  # AdvancedProfiler(),#
        track_grad_norm=con.track_grad_norm,
        check_val_every_n_epoch=con.check_val_n_epoch,  # Uses a complex logic to combine above param with this.
        weights_summary=con.weight_summary,  # Whether to print weights summary
        logger=logger,
        default_save_path=con.ckpt_dir,
        callbacks=[PrintingCallback()],
        resume_from_checkpoint=con.ckpt,  # Currently unused
        progress_bar_refresh_rate=con.progress_bar,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=con.grad_clip,

    )
    # TODO: trainer sets deterministic to False by default. Why is code deterministic?
    trainer.fit(model)
    # Functionality to save last checkpoint. AUtomatically saves even if program is quit from wandb!
    if con.save_last:
        filepath = os.path.join("CLAD-pytorch", "version" + str(con.seed) +
                                "_" + str(model.logger.version), str(model.current_epoch) + '.ckpt')
        # Creating log dirs if they don't exist.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print("Log dir: ", os.path.join("CLAD-pytorch",
                                        "version" + str(con.seed) +
                                        "_" + str(model.logger.version)))
        trainer.save_checkpoint(filepath=filepath)
    # print(trainer.profiler.summary())


if __name__ == '__main__':
    main()
