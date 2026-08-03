# original refer to: https://github.com/lujiarui/Str2Str/blob/main/src/utils/checkpoint_utils.py
# usage refer to: https://github.com/lujiarui/Str2Str/blob/main/src/train.py#L86

import torch

def load_model_checkpoint(model, ckpt_path):
    """Load state dict from checkpoint file.

    :param model: The model to load the state dict into.
    :param ckpt_path: The path to the checkpoint file.
    """
    if ckpt_path is None: 
        return model, None 
    
    # The ckpt_path ending with .ckpt is a checkpoint file saved by pytorch-lightning.
    # If the ckpt_path is a .pth file, it is viewed as a checkpoint file saved by pytorch
    # such that only net parameters are loaded. 
    # (This may avoid the ambiguity of loading #epochs/lr for finetuning)
    if ckpt_path.endswith(".pth"):  
        # ⚠️ remember to add `weights_only=False` for pytorch2.6+ version, but not needed for pytorch2.5 and below
        net_params = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)['state_dict']
        net_params = {k.replace('net.', ''): v for k, v in net_params.items()}
        model.net.load_state_dict(net_params)
        ckpt_path = None
    elif ckpt_path.endswith(".ckpt"):
        # will be handled later by the trainer
        pass
    else:
        # suffix check
        raise ValueError(f"ckpt_path {ckpt_path} is not a valid checkpoint file.")
    
    return model, ckpt_path