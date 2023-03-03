# Prismer

This repository contains the source code of **Prismer** and **PrismerZ** from the paper, Prismer: A vision-Language Model with An Ensemble of Experts.

Please check out our project page, including more interesting and behind-the-scenes dicussions.

<img src="helpers/intro.png"  width="80%"/>

## Get Started
The implementation is based on `PyTorch 1.13`, and built on with Huggingface [`accelerate`](https://github.com/huggingface/accelerate) for highly readable and optimised multi-node multi-gpu training.

First, let's install all package dependencies.
```bash
pip install -r requirements.txt
```

## Datasets

### Pre-training
We pre-train Prismer/PrismerZ with a combination of five widely used image-alt/text datasets, with downloadable pre-organised data lists provided below.
- COCO 2014: The Karpathy training split (which will also used for fine-tuning).
- Visual Genome: The official Visual Genome Captioning set.
- CC3M + SGU: Pre-filtered and re-captioned by BLIP.
- CC12M: Pre-filtered and re-captioned by BLIP.

To download web datasets (CC3M, SGU, CC12M), it is highly recommended to use [img2dataset](https://github.com/rom1504/img2dataset), a highly optimised tool for large-scale web scraping. An example bash script of using `img2dataset` to download `cc12m` dataset is provided below. 
```bash
img2dataset --url_list filtered_cc12m.json --input_format "json" --url_col "url" --caption_col "caption" --output_folder cc12m --processes_count 16 --thread_count 64 --image_size 256
```

### Image Captioning / VQA
We evaluate image captioning performance on two datasets, COCO 2014 and NoCaps; and VQA performance on VQAv2 dataset. In VQA tasks, we additionally augment the training data with Visual Genome QA. Similarly, we prepare and organise the training and evaluation data list following BLIP.

- Image Captioning: Including COCO (Karpathy Split) and NoCAPS.
- VQAv2: Including VQAv2 and VG QA.

## Generating Expert Labels
Before starting any experiments with Prismer, we need to pre-generate the expert labels, as part of the datasets. In `experts` folder, we have included all 6 experts we introduced in our original paper. We have organised each expert's codebase with a shared and simple APIs.

Specifically for segmentation experts, please first install deformable convolution by `cd experts/segmentation/mask2former/modeling/pixel_decoder/ops` and `sh make.sh`.

And then to generate each expert labels, simply edit the `configs/experts.yaml` with the corresponding data paths, and run the following commands.
```bash
export PYTHONPATH=.
python experts/generate_{EXPERT_NAME}.py
```
*Note: Expert label generation is only required for Prismer models, not for PrimerZ models.*

## Experiments
We have provided both Prismer and PrimerZ for pre-training checkpoints (for zero-shot image captioning), as well as fined-tuning checkpoints on VQAv2 and COCO datasets. With these checkpoints, it should be expected to reproduce the exact performance listed below.

| Method         | Pre-trained [Zero-shot] | COCO [Fine-tuned]   | VQAv2 [Fine-tuned] |
|----------------|-------------------------|---------------------|-------------------|
| PrismerZ-BASE  | COCO CIDEr [109.6]      | COCO CIDEr [133.7]	 | test-dev [76.58]  |
| Prismer-BASE   | COCO CIDEr [122.6]      | COCO CIDEr [135.1]	 | test-dev [76.84]  |
| PrismerZ-LARGE | COCO CIDEr [124.8]      | COCO CIDEr [135.7]	 | test-dev [77.49]  |
| Prismer-LARGE  | COCO CIDEr [129.7]      | COCO CIDEr [136.5]	 | test-dev [78.42]  |

All checkpoints can be downloaded in this folder.

*Note: COCO captioning performance with pre-trained model checkpoints are evaluated in a zero-shot setting.*

### Prepare Accelerator Config
We have provided a script to generate the corresponding `accelerate` training configs with a minimal effort. For both single-node multi-gpu and multi-node multi-gpu training, simply run the following commands,
```bash
# To get your machine rank 0 IP address
hostname -i
# And for each machine, run the following command, set --num_machines 1 in a single-node setting
python generate_config.py —-main_ip {MAIN_IP} -—rank {MACHINE_RANK} —-num_machines {TOTAL_MACHINES}
```


*Note: Remember to install java by `sudo apt-get install default-jre` to evaluate COCO's captioning performance.*

### Evaluation
After setting up the `accelerate` configs, to evaluate the model checkpoints, please run
```bash
# Zero-shot Image Captioning (Remember to remove caption prefix in the config files)
python train_caption.py --exp_name {MODEL_NAME} --evaluate
# Fine-tuned Image Captioning
python train_caption.py --exp_name {MODEL_NAME} --from_checkpoint --evaluate
# Fine-tuned VQA
python train_vqa.py --exp_name {MODEL_NAME} --from_checkpoint --evaluate
```

### Training / Fine-tuning
Similarly, to pre-train or fine-tune any model checkpoints, please run
```bash
# To train/fine-tuning from scratch
python train_{TASK}.py --exp_name {MODEL_NAME}
# To train/fine-tuning from the latest checkpoints (saved every epoch)
python train_{TASK}.py --exp_name {MODEL_NAME} --from_checkpoint 
```
We have also provided model sharding tools by using PyTorch's official FSDP toolkit. With the same training commands, further add `--shard_grad_op` for ZeRO-2 Sharding (Gradients + Optimiser States), or `--full_shard` for ZeRO-3 Sharding  (ZeRO-2 + Network Parameters). 

*Note: You should expect the error range for VQAv2 Acc. to be less than 0.1; for COCO/NoCAPs CIDEr to be less than 1.0.*

## Citation

If you found this code/work to be useful in your own research, please considering citing the following:


```bibtex
@article{liu2023prismer,
    title={Prismer: A Vision-Language Model with An Ensemble of Experts},
    author={Liu, Shikun and Fan, Linxi and Johns, Edward and Xiao, Chaowei and Yu, Zhiding and Anandkumar, Anima},
    journal={arXiv preprint arXiv:TODO},
    year={2023}
}
```

## License
Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. 

The model checkpoints are shared under CC-BY-NC-SA-4.0. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

## Acknowledgement
We would like to thank all the authors open-sourcing their pre-trained models to make this project possible.

## Contact
If you have any questions, please contact `sk.lorenmt@gmail.com`.