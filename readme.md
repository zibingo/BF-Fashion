# BF-Fashion

![BF-Fashion](FrameWork.jpg)

# TODO LIST

- [x] processed datasets
- [x] training code
- [x] inference code
- [x] demo model

# 1.Environment

```sh
git clone https://github.com/zibingo/BF-Fashion.git
cd BF-Fashion
conda env create -f environment.yaml
conda activate ldm
pip install -e .
cd src/taming-transformers
pip install -e .
cd ../..
```

download https://ommer-lab.com/files/latent-diffusion/vq-f8.zip and unzip it to the "models\first_stage_models\vq-f8"

# 2.Data preparation

Download the [Cleaned Type-aware Dataset](https://github.com/AemikaChow/AiDLab-fAshIon-Data/blob/main/Datasets/cleaned-type.md) and apply [HED](https://github.com/s9xie/hed) to generate sketches.

You can also download our [processed dataset](https://drive.google.com/file/d/1E5HW_17IfjfkBlEPLCaE6mbjPtWvJU2H/view?usp=drive_link) to the root of your project folder.

# 3.Train

```sh
sh run_train.sh
```

# 4.Inference

```sh
sh run_batch_sample.sh
```

# 5.Evaluation

```sh
python m_class_fid_lpips.py
python m_class_histo_loss.py
```

