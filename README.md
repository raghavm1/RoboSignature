# RoboSignature: Robust Signature and Watermarking on Diverse Image Attacks

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

## Acknowledgements

This code is based on the following repositories:

- https://github.com/facebookresearch/stable_signature
- https://github.com/Stability-AI/stablediffusion
- https://github.com/SteffenCzolbe/PerceptualSimilarity

To train the watermark encoder/extractor, you can also refer to the following repository https://github.com/ando-khachatryan/HiDDeN.


## Setup
### Requirements

To install the main dependencies, we recommand using conda.
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the remaining dependencies with pip:
```cmd
pip install -r requirements.txt
```

### Models and data

#### Data

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

## Usage

### Fine-tune LDM decoder

This code is specified in `slurmscript` in the root directory, which is a batch file being used on an HPC. This file can be used as a sample to finetune the LDM decoder.

### Tamper-Resistant finetuning 

TODO

### Decode and Evaluate

The `decode.ipynb` notebook contains a full example of the decoding and associated statistical test.

The `run_eval.py` script can be used to get the robustness and quality metrics on a folder of images.
For instance:
```
python run_eval.py --eval_imgs False --eval_bits True \
    --img_dir path/to/imgs_w \
    --key_str '111010110101000001010111010011010100010000100111'
```
will return a csv file containing bit accuracy for different attacks applied before decoding.

```
python run_eval.py --eval_imgs True --eval_bits False \
    --img_dir path/to/imgs_w --img_dir_nw path/to/imgs_nw 
```
will return a csv file containing image metrics (PSNR, SSIM, LPIPS) between watermarked (`_w`) and non-watermarked (`_nw`) images.

## License

The majority of this code is licensed under CC-BY-NC, however portions of the project are available under separate license terms: `src/ldm` and `src/taming` are licensed under the MIT license.

## Citations

```
@article{fernandez2023stable,
  title={The Stable Signature: Rooting Watermarks in Latent Diffusion Models},
  author={Fernandez, Pierre and Couairon, Guillaume and J{\'e}gou, Herv{\'e} and Douze, Matthijs and Furon, Teddy},
  journal={ICCV},
  year={2023}
}
```
