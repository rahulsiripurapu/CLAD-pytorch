import os

import wandb
import numpy as np
from pytorch_lightning import Callback


class PrintingCallback(Callback):
    #TODO Make it inherit PrintingCallback from SAC

    def on_train_start(self, trainer, pl_module):
        if pl_module.hparams.track_grad_norm:
            #TODO why does printcallback have model logging?
            print("Tracking grad norm using wandb")
            pl_module.logger.watch(pl_module.policy,log="all")
            pl_module.logger.watch(pl_module.vf, log="all")
            pl_module.logger.watch(pl_module.qf, log="all")
            # pl_module.logger.experiment.monitor() #TODO check if using monitor is more convenient and elegant

    def on_sanity_check_start(self, trainer, pl_module):
        print(pl_module.modules)
        pl_module.policy._device = pl_module.hparams.device
        # print("pz device: ",pl_module._p_z.device)
        if(pl_module.on_gpu):
            pl_module._p_z = pl_module._p_z.cuda(pl_module.hparams.device)
        # print("p_z: ",pl_module._p_z.device)
        # TODO: move this elsewhere. If validation is disabled policy will not be sent to GPU
        # pl_module.pool.add_samples(pl_module.sampler.sample(pl_module.hparams.min_pool_size,pl_module.policy))

    def on_validation_end(self, trainer, pl_module):
        print("\nvalidation path return: ",pl_module.val_path_return)
        # print("pool size: ", pl_module.pool._size)
        # print("pool reward avg: ",np.mean(pl_module.pool._rewards[:pl_module.pool._size]))
        # print("pool reward max: ", np.max(pl_module.pool._rewards[:pl_module.pool._size]))
        # print("pool reward min: ", np.min(pl_module.pool._rewards[:pl_module.pool._size]))
        #TODO why is checkpoiting happening in printcallback?
        #TODO better way to check wandb than dir attribute?
        # if(pl_module.hparams.enable_checkpointing):
        #     if(hasattr(pl_module.logger.experiment,"dir")):
        #         trainer.save_checkpoint(os.path.join(pl_module.logger.experiment.dir,
        #                                              "latest" + str(pl_module.global_step) + '.ckpt'))
        #         print("\nCheckpointing model: ",os.path.join(pl_module.logger.experiment.dir,
        #                                              "latest" + str(pl_module.global_step) + '.ckpt'))
        #     else:
        #         #TODO may need to use os.path.join in other places as well
        #         print("Log dir: ",os.path.join(pl_module.logger.save_dir,
        #                                        pl_module.logger.name,
        #                                        "version_" + str(pl_module.logger.version),
        #                                        "checkpoints"))
        #         trainer.save_checkpoint(os.path.join(pl_module.logger.save_dir,
        #                                              pl_module.logger.name, "version_" +
        #                                              str(pl_module.logger.version),
        #                                              "checkpoints",
        #                                              "latest" + str(pl_module.global_step) + '.ckpt'))
        #TODO use log_model flag in wandb to save ckpts to cloud

        # wandb.save('latest.ckpt')
        os.environ['WANDB_IGNORE_GLOBS'] += '*.mp4'
        if pl_module.hparams.render_validation:
            if hasattr(pl_module.logger.experiment, "dir"):
                # _save_video(pl_module.ims,os.path.join(pl_module.logger.experiment.dir,
                #                                      "latest" + str(pl_module.global_step) + '.mp4'))
                _save_video(pl_module.ims, pl_module.hparams.path_name + str(pl_module.z) + '.mp4')
            else:
                #TODO: use better file naming than "version" here and above
                _save_video(pl_module.ims, os.path.join(pl_module.logger.save_dir,
                                                     pl_module.logger.name, "version_" +
                                                     str(pl_module.logger.version),
                                                     "videos",
                                                     "latest" + str(pl_module.global_step) + '.mp4'))

        #     video_array = np.asarray(pl_module.ims)
        #     print("\nUploading video shape: ",video_array.shape)
        #     pl_module.logger.experiment.log({"video": wandb.Video(video_array, fps=4, format="gif")})

    # def on_batch_end(self, trainer, pl_module):
    #     if(pl_module.on_gpu):
    #         pl_module.logger.experiment.summary['q_values'] = pl_module.q_values.cpu()
    #         pl_module.logger.experiment.summary['values'] = pl_module.values.cpu()
    #     else:
    #         pl_module.logger.experiment.summary['q_values'] = pl_module.q_values #TODO: is else needed, cpu might be redundant
    #         pl_module.logger.experiment.summary['values'] = pl_module.values

    #TODO record video and save file
    # def on_train_end(self, trainer, pl_module):
    #     #TODO doesn't save log file when not using wandb.
    #     if not (hasattr(pl_module.logger.experiment, "dir")):
    #         pl_module.logger.save()

    # def on_batch_end(self, trainer, pl_module):
    #     print(pl_module.batch_idx)
    #     pl_module.

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def _save_video(ims, filename):
    import cv2
    # assert all(['ims' in path for path in paths])
    # ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

    # TODO: close env after experiment ends