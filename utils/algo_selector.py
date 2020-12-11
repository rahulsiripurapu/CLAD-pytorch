import torch
from copy import deepcopy
from reward_functions.discriminator import Discriminator


def algo_selector(con):
    if con.algo == 'SAC':
        if con.version == 'openai':
            print("\nUsing OpenAI version\n")
            from algos.sac_v2 import SAC
            model = SAC(con)
        else:
            from algos.sac import SAC
            model = SAC(con)
    elif con.algo == 'DIAYN':
        if con.version == 'openai':
            print("\nUsing OpenAI version\n")
            from algos.diayn_v2 import DIAYN
            model = DIAYN(con)
        else:
            from algos.diayn import DIAYN
            model = DIAYN(con)
    elif con.algo == 'DIAYN_finetune':
        if con.version == 'openai':
            print("\nUsing OpenAI version\n")
            from algos.diayn_finetune_v2 import DIAYN_finetune
        else:
            from algos.diayn_finetune import DIAYN_finetune
        model = DIAYN_finetune(con)
        if con.ckpt_load:
            # model.load_from_checkpoint(con.ckpt_load) # Does not work due to hparams issue
            # hence using below workaround
            checkpoint = torch.load(con.ckpt_load, map_location=torch.device(con.device))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("Loaded model from checkpoint: ", con.ckpt_load, "\nContaining: ", checkpoint['state_dict'].keys())
    elif con.algo == 'HERF':
        from algos.diayn_herf import DHERF
        model = DHERF(con)
        if con.ckpt_load:
            checkpoint = torch.load(con.ckpt_load, map_location=torch.device(con.device))
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("Loaded model from checkpoint: ", con.ckpt_load, "\nContaining: ", checkpoint['state_dict'].keys())
    elif con.algo == 'DIAYN_retrain':
        from algos.diayn_retrain import DIAYN_retrain
        model = DIAYN_retrain(con)
        # Commented code to ensure that retrain uses same seeded inits as original DIAYN in case skills may improve
        # Didnt help so commented out.
        # qf = deepcopy(model.qf.state_dict())
        # vf = deepcopy(model.vf.state_dict())
        # policy = deepcopy(model.policy.state_dict())
        assert con.ckpt_load
        checkpoint = torch.load(con.ckpt_load, map_location=torch.device(con.device))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if "distiller0" in checkpoint.keys():
            for i in range(len(con.disc_size)):
                # Loads distillers in reverse order due to mistake in distill config which trains largest as first.
                model.distiller[i].load_state_dict((checkpoint['distiller%d'%(len(con.disc_size)-1-i)]))
        print("Loaded model from checkpoint: ", con.ckpt_load, "\nContaining: ", checkpoint['state_dict'].keys(), checkpoint.keys())
        # model.policy.load_state_dict(policy)
        # model.qf.load_state_dict(qf)
        # model.vf.load_state_dict(vf)
        # model.vf_target.load_state_dict(vf)
        # print("Reloaded original init state dicts")
        """checkpoint['state_dict'].keys() odict_keys(['qf.mlp.net.0.weight', 'qf.mlp.net.0.bias',
        'qf.mlp.net.2.weight', 'qf.mlp.net.2.bias', 'qf.mlp.net.4.weight', 'qf.mlp.net.4.bias',
        'vf.mlp.net.0.weight', 'vf.mlp.net.0.bias', 'vf.mlp.net.2.weight', 'vf.mlp.net.2.bias',
        'vf.mlp.net.4.weight', 'vf.mlp.net.4.bias', 'vf_target.mlp.net.0.weight', 'vf_target.mlp.net.0.bias',
        'vf_target.mlp.net.2.weight', 'vf_target.mlp.net.2.bias', 'vf_target.mlp.net.4.weight',
        'vf_target.mlp.net.4.bias', 'policy.net.0.weight', 'policy.net.0.bias', 'policy.net.2.weight',
        'policy.net.2.bias', 'policy.net.4.weight', 'policy.net.4.bias', 'policy._qf.mlp.net.0.weight',
        'policy._qf.mlp.net.0.bias', 'policy._qf.mlp.net.2.weight', 'policy._qf.mlp.net.2.bias',
        'policy._qf.mlp.net.4.weight', 'policy._qf.mlp.net.4.bias', 'discriminator.mlp.net.0.weight',
        'discriminator.mlp.net.0.bias', 'discriminator.mlp.net.2.weight', 'discriminator.mlp.net.2.bias',
        'discriminator.mlp.net.4.weight', 'discriminator.mlp.net.4.bias']) checkpoint.keys() dict_keys(['epoch',
        'global_step', 'checkpoint_callback_best', 'optimizer_states', 'lr_schedulers', 'state_dict', 'hparams',
        'hparams_type'])

        """
    elif con.algo == 'DIAYN_distill':
        # assert 0 == 1  # Fix inverted distiller in retrain first!
        from algos.diayn_distill import DIAYN_distill
        model = DIAYN_distill(con)
        assert con.ckpt_load
        checkpoint = torch.load(con.ckpt_load, map_location=torch.device(con.device))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Loading replay buffer from checkpiont if needed to distill new ckpts from the replay buffer
        if con.load_pool:
            model.pool = checkpoint['pool']
        print("Loaded model from checkpoint: ", con.ckpt_load, "\nContaining: ", checkpoint['state_dict'].keys())
    else:
        from algos.diayn_viz import DIAYN_viz
        model = DIAYN_viz(con)
        assert con.ckpt_load
        # ckpt = torch.load(
        #     "/Users/rahulsiripurapu/CLAD/ckpts-visualize/half-cheetah/DIAYN/seed3-finaldiscriminator/epoch=990.ckpt",
        #     map_location="cpu")
        # model.load_state_dict(ckpt['state_dict'], strict=False)
        # discdict = model.discriminator.state_dict()
        # model.discriminator = Discriminator(model.Do - model._num_skills, [150,150],
        #                                model._num_skills)
        checkpoint = torch.load(con.ckpt_load, map_location=torch.device(con.device))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # model.discriminator = Discriminator(model.Do - model._num_skills, [300, 300],
        #                                     model._num_skills)
        # model.discriminator.load_state_dict(discdict)
        print("Loaded model from checkpoint: ", con.ckpt_load, "\nContaining: ", checkpoint['state_dict'].keys())

    return model
