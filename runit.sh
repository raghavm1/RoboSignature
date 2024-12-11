#!/bin/bash

export PYTHONPATH="/scratch/rm6418/Tamper_Resistant_Stable_Signature/src/:$PYTHONPATH"
source /ext3/env.sh
python Tamper_Resistant_Stable_Signature/tamper_resistant_training.py --num_keys 1 \
    --ldm_config Tamper_Resistant_Stable_Signature/stable-diffusion-2-1/v2-inference.yaml \
    --ldm_ckpt Tamper_Resistant_Stable_Signature/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt \
    --msg_decoder_path Tamper_Resistant_Stable_Signature/models/dec_48b_whit.torchscript.pt \
    --train_dir Tamper_Resistant_Stable_Signature/train2014500/ \
    --val_dir Tamper_Resistant_Stable_Signature/test2014/  \
    --atrain_dir Tamper_Resistant_Stable_Signature/train2014_10000/ \
    --finetuned_ckpt /scratch/rm6418/output/checkpoint_000.pth \
    --strategy 1 \
    --inner_steps 2 \
    --outer_steps 5 \
    --steps 1
	#2> /scratch/rm6418/Tamper_Resistant_Stable_Signature/log.err > /scratch/rm6418/Tamper_Resistant_Stable_Signature/log.out
