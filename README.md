## Online Unsupervised Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions

Source code for "Online Unsupervised Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions", ECCV 2022.
This is the code has been implemented to perform training and evaluation of UDA approaches in continuous scenarios. 
The library has been implemented in PyTorch 1.7.1. Some newer versions should work as well.

![Method Cover](assets/images/cover-min.png)

**All assets to run a simple inference can be found** [here](https://drive.google.com/drive/folders/14X3XUjvnl0gwML4k7FI1yB9u9-oQmo-x?usp=sharing)

Moreover, recording and tracking for the run is happening through [wandb](https://wandb.com) if you haven't an account is necessary to track the adaptation.

## Repositories

We would advise you to use conda or miniconda to run the package. 
Run the following command to install the necessary modules:

```
conda env create -f environment.yml
```
<!-- Take note the `requirements.txt` contains absolute versions, newer versions might also be compatible with the code. -->
After creating the environment, load it using  `conda activate ouda`.

You would then need to login to wandb to record the experiments simply type `wandb login`.

## Creating the rainy dataset
First download the Cityscapes dataset from [here](https://www.cityscapes-dataset.com/).
To add rain to the cityscapes dataset you need to follow the steps as shown [here](https://team.inria.fr/rits/computer-vision/weather-augment/). The autors provide the rain mask for each image. With their dev-kit one can create the rainy images. Moreover, for the validation it is possible to create them as described [here](https://github.com/cv-rits/rain-rendering/issues/3). We are in talks with the authors to make the creation of the dataset easier.

## Download the pretrained source model and prototypes

Download the files `precomputed_prototypes.pickle` , `pretrained_resnet50_miou645.pth` and save them into a folder named `pretrained`

## Edit configuration

Open the file `configs/hybrid_switch.yml` and edit the `PATH` variable with the location of the dataset. The path should point to the leftImg8bit and gtFine folders. Make sure that the paths for the pretrained models at `METHOD.ADAPTATION.PROTO_ONLINE_HYBRIDSWITCH.LOAD_PROTO` and `MODEL.LOAD` are correct. The paths should point to the pretrained source and prototypes downloaded in the previous steps.

## Run

We recommend using a powerful graphics card with at least 16GB of VRAM. To run this code it needs a bit over 1 day in an RTX3090. If necessary one can play arround with the batch size and resolution on the configuration file to test the approach, but results will not be replicated.

To run first one should initialise wandb `wandb login` and then simply run `python train_ouda.py --cfg=configs/hybrid_switch.yml`

The run performs evaluation accross domains from the start and for each pass through the data. We demonstrated how to run the hybrid switch but by configuring or selecting other configuration files one can use different switches or approaches. By default the approach will create folders to save predictions.

## Code library

The approaches can be found under `framework/domain_adaptation/methods`:
The code that handles the prototypes can be found in: `framework/domain_adaptation/methods/prototype_handler.py`
While the switching approach is written here: `framework/domain_adaptation/methods/prototypes.py`
The Confidence switch (and Soft) is here: `framework/domain_adaptation/methods/prototypes_hswitch.py`
The Confidence Derivative Switch is here: `framework/domain_adaptation/methods/prototypes_vswitch.py`
Lastly the code for the hybrid switch can be found here: `framework/domain_adaptation/methods/prototypes_hybrid_switch.py`
Advent is the implementation here: `framework/domain_adaptation/methods/advent_da.py`

## Citation

If you find this repo useful for your work, please cite our paper:

```shell
@inproceedings{Panagiotakopoulos_ECCV_2022,
  title     = {Online Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions},
  author    = {Panagiotakopoulos, Theodoros and
               Dovesi, Pier Luigi and
               H{\"a}renstam-Nielsen, Linus and
               Poggi, Matteo},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```   

## Regards

Don't hesitate to contact us if there are questions about the code or about the different options in the cfg file.
Thank you!!










