# RoboSignature: Robust Signature and Watermarking on Diverse Image Attacks

Our paper is on [arxiv](https://arxiv.org/abs/2412.19834)!

## Dependencies

To install the main dependencies, we recommand using conda.
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the remaining dependencies with pip:
```cmd
pip install -r requirements.txt
```

## Setup

Download test set images

```cmd
wget http://images.cocodataset.org/zips/test2014.zip
unzip test2014.zip
rm test2014.zip
```

Download checkpoint model

```cmd
wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt -P stable-diffusion-2-1-base/
```

## Models and data

### Data

Similar to the Stable Signature paper, this paper uses the [COCO](https://cocodataset.org/) dataset to fine-tune the LDM decoder (we filtered images containing people).
All you need is around 500 images for fine-tuning the LDM decoder (preferably over 256x256).

#### Watermark models

The watermark extractor model can be downloaded in the following link.

| Model | Checkpoint |
| --- | --- |
| Extractor | [dec_48b.pth](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b.pth) |


#### Stable Diffusion models

Create LDM configs and checkpoints from the [Hugging Face](https://huggingface.co/stabilityai) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion) repositories.
The code should also work for Stable Diffusion v1 without any change. 
For other models (like old LDMs or VQGANs), you may need to adapt the code to load the checkpoints.

## Acknowledgements

This code is based on the following repositories:

- https://github.com/facebookresearch/stable_signature
- https://github.com/Stability-AI/stablediffusion
- https://github.com/SteffenCzolbe/PerceptualSimilarity

To train the watermark encoder/extractor, you can also refer to the following repository https://github.com/ando-khachatryan/HiDDeN.

## Usage

### Fine-tune LDM decoder

The script is specified in `slurmscript` in the root directory, which is a batch file being used on an HPC. This file can be used as a sample to finetune the LDM decoder.

```cmd
python Tamper_Resistant_Stable_Signature/finetune_ldm_decoder.py --num_keys 1 \
    --ldm_config Tamper_Resistant_Stable_Signature/stable-diffusion-2-1/v2-inference.yaml \
    --ldm_ckpt Tamper_Resistant_Stable_Signature/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt \
    --msg_decoder_path Tamper_Resistant_Stable_Signature/models/dec_48b_whit.torchscript.pt \
    --train_dir Tamper_Resistant_Stable_Signature/train2014500/ \
    --val_dir Tamper_Resistant_Stable_Signature/test2014/
```

### Tamper-Resistant finetuning 

```cmd
python Tamper_Resistant_Stable_Signature/tamper_resistant_training.py --num_keys 1 \
    --ldm_config Tamper_Resistant_Stable_Signature/stable-diffusion-2-1/v2-inference.yaml \
    --ldm_ckpt Tamper_Resistant_Stable_Signature/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt \
    --msg_decoder_path Tamper_Resistant_Stable_Signature/models/dec_48b_whit.torchscript.pt \
    --train_dir Tamper_Resistant_Stable_Signature/train2014500/ \
    --val_dir Tamper_Resistant_Stable_Signature/test2014/ \
    --atrain_dir Tamper_Resistant_Stable_Signature/train2014_10000/ \
    --finetuned_ckpt /scratch/gb2762/output/checkpoint_000.pth \
    --strategy 1 \
    --inner_steps 50 \
    --outer_steps 100 \
    --steps 100"
```

## License

The majority of this code is licensed under CC-BY-NC, however portions of the project are available under separate license terms: `src/ldm` and `src/taming` are licensed under the MIT license.

## Citations

If you like our work, consider giving us a star and citing our paper as - 

```
@misc{shaan2024robosignaturerobustsignaturewatermarking,
      title={RoboSignature: Robust Signature and Watermarking on Network Attacks}, 
      author={Aryaman Shaan and Garvit Banga and Raghav Mantri},
      year={2024},
      eprint={2412.19834},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2412.19834}, 
}
```
