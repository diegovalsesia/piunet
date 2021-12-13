# Permutation invariance and uncertainty in multitemporal image super-resolution

PIUnet is a novel Multi Image Super-Resolution (MISR) deep neural network that exploits both spatial and temporal correlations to recover a single high resolution image from multiple unregistered low resolution images. It is fully invariant to the ordering of the input images and produces an estimate of the aleatoric uncertainty together with the super-resolved image.

This repository contains the pytorch implementation of PIUnet, trained and tested on the PROBA-V dataset provided by ESA’s [Advanced Concepts Team](http://www.esa.int/gsp/ACT/index.html) in the context of the [European Space Agency's Kelvin competition](https://kelvins.esa.int/proba-v-super-resolution/home/). 


BibTex reference:
```
@article{valsesia2021piunet,
  title={Permutation invariance and uncertainty in multitemporal image super-resolution},
  author={Valsesia, Diego and Magli, Enrico},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={},
  number={},
  pages={},
  year={2021},
  publisher={IEEE}
}
```

#### Setup to get started
Make sure you have Python3 and all the required python packages installed:
```
pip install -r requirements.txt
```

The code has been tested on Pytorch 1.7.1 for CUDA 10.1. Newer version will likely work as well.

#### Load data from Kelvin Competition and create the training set and the validation set
- Download the PROBA-V dataset from our train/validation partitioned version ([probav_data.zip](https://www.dropbox.com/s/0724byclle90nhg/probav_data.zip?dl=0)) and extract it under _./Dataset/probav\_data_
- Load the dataset from the directories and save it to pickles by running _Save\_dataset\_pickles.ipynb_ notebook
- Run the _preprocessing\_dataset.ipynb_ notebook to create training dataset and validation dataset for both NIR and RED bands

#### Usage
- The _config.py_ contains your network configuration before starting training the model:

```
"N_feat" : no. of feature channels (42)
"R_bneck" : no. of features in attention bottlenecks (8)
"N_tefa": no. of TEFA blocks (16),
"N_heads": no. of heads for multihead attention (1),
"patch_size": training patch size (32),
"batch_size": training batch size (24),
"N_epoch": no. of training epochs (200),
"learning_rate": optimizer learning rate (1e-4),
"train_lr_file": path to npy file with train LR images,
"train_hr_file": path to npy file with train HR images,
"train_masks_file": path to npy file with train image masks,
"val_lr_file": path to npy file with validation LR images,
"val_hr_file": path to npy file with validation HR images,
"val_masks_file": path to npy file with validation image masks,
"max_train_scenes": maximum no. of images to be used for training (full dataset: 563 for NIR, 591 for RED)
```

- Train the model by running the training script:
```
./launcher_train.sh
```
OR (after making sure the supplied log_dir and save_dir exist)
```
CUDA_VISIBLE_DEVICES=0 python Code/piunet/main.py --log_dir log_dir/piunet/ --save_dir Results/piunet/
```

Default parameters require approximately 18GB of GPU memory for training.

#### Pretrained models
- Pretrained model on training data (approximately journal results): [NIR](https://www.dropbox.com/s/hme35p9p3lrfj4s/nir_model_checkpoint.pt?dl=0), [RED](https://www.dropbox.com/s/sl5wgph3o86r7ah/red_model_checkpoint.pt?dl=0)
 

#### Testing
- Run the _testing\_notebook.ipynb_ notebook to run a trained model on some test data and look at the results.


## Authors & Contacts
PIUnet is based on work by the [Image Processing and Learning](https://ipl.polito.it/) group of Politecnico di Torino. Contacts: Diego Valsesia (diego.valsesia AT polito.it), Enrico Magli (enrico.magli AT polito.it).


