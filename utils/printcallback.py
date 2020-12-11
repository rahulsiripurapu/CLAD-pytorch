import os

import wandb
import numpy as np
from pytorch_lightning import Callback
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PrintingCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        if pl_module.hparams.track_grad_norm:
            # TODO why does printcallback have model logging?
            print("Tracking grad norm using wandb")
            pl_module.logger.watch(pl_module.policy, log="all")
            pl_module.logger.watch(pl_module.vf, log="all")
            pl_module.logger.watch(pl_module.qf, log="all")
            # pl_module.logger.experiment.monitor() #TODO check if using monitor is more convenient and elegant

    def on_sanity_check_start(self, trainer, pl_module):
        print(pl_module.modules)
        # pl_module.policy._device = pl_module.hparams.device
        # TODO: Remove this, policy doesn't need to be manually sent anymore: verify.
        # pl_module.pool.add_samples(pl_module.sampler.sample(pl_module.hparams.min_pool_size,pl_module.policy))

    def get_label(self,i,algo):
        if algo=="HERF":
            skilllist = ["r4","r3","r2","r1","rb1","rb2","rb3","rb4","f4","f3","f2","f1","fb4","fb3","fb2","fb1","s90",
                         "s60","s120","b90","b60","jf90","jf60","jb90","jb60","2r4","2r3","2r2","2r1","2rb1","2rb2",
                         "2rb3","2rb4","2f4","2f3","2f2","2f1","2fb4","2fb3","2fb2","2fb1","2s90","2s60","2s120",
                         "2b90","2b60","2jf90","2jf60","2jb90","2jb60"]

            return skilllist[i]
        else:
            return str(i)

    def on_validation_end(self, trainer, pl_module):
        print("\nvalidation path return: ", pl_module.val_path_return)
        if (pl_module.hparams.algo == "DIAYN_retrain" or pl_module.hparams.algo=="DIAYN_distill"
            or pl_module.hparams.algo=="HERF") and hasattr(pl_module.logger.experiment, "dir") and pl_module.current_epoch%10==0:
            fig, ax = plt.subplots()
            for i in range(pl_module._num_skills):
                if len(pl_module.skilldata_val[i]) > 0:
                    data = np.asarray(pl_module.skilldata_val[i])
                    ax.plot(data[:, -1], np.mean(data[:, :-1], axis=-1), label=self.get_label(i,pl_module.hparams.algo))
            ax.set(xlabel="Epoch", ylabel="Reward")
            ax.set_title("Mean reward per skill")
            pl_module.logger.experiment.log({"Reward ": plt})
            plt.close()
            # pl_module.logger.experiment.log({'heatmap_with_text': wandb.plots.HeatMap(np.arange(pl_module._num_skills), np.arange(pl_module.hparams.max_epochs), pl_module.reward_map.T, show_text=False)})

        # print("pool size: ", pl_module.pool._size)
        # print("pool reward avg: ",np.mean(pl_module.pool._rewards[:pl_module.pool._size]))
        # print("pool reward max: ", np.max(pl_module.pool._rewards[:pl_module.pool._size]))
        # print("pool reward min: ", np.min(pl_module.pool._rewards[:pl_module.pool._size]))
        # TODO why is checkpoiting happening in printcallback?
        # TODO better way to check wandb than dir attribute?
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
        # TODO use log_model flag in wandb to save ckpts to cloud

        # wandb.save('latest.ckpt')
        # TODO put all common stuff in another file in utils and reuse those here.
        os.environ['WANDB_IGNORE_GLOBS'] += '*.mp4'
        if pl_module.hparams.render_validation:
            if hasattr(pl_module.logger.experiment, "dir"):
                # _save_video(pl_module.ims,os.path.join(pl_module.logger.experiment.dir,
                #                                      "latest" + str(pl_module.global_step) + '.mp4'))
                _save_video(pl_module.ims, pl_module.hparams.path_name + str(pl_module.z) + '.mp4')
            else:
                # TODO: use better file naming than "version" here and above
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

    # TODO record video and save file
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
