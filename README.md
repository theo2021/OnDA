## Online Unsupervised Domain Adaptation for Semantic Segmentation in Ever-Changing Conditions

This is the code has been implemented to perform training and evaluation of UDA approaches in continuous scenarios. The library has been implemented in PyTorch 
1.7.1. Some newer versions should work as well.

**All assets to run a simple inference can be found** [here](https://drive.google.com/drive/folders/14X3XUjvnl0gwML4k7FI1yB9u9-oQmo-x?usp=sharing)

Moreover, recording and tracking for the run is happening through [wandb](https://wandb.com) if you haven't an account is necessary to track the adaptation.

## Repositories

Run the following command to install the necessary modules:

```
pip install -r requirements.txt
```
Take note the `requirements.txt` contains absolute versions, newer versions might also be compatible with the code.

## Downloading the dataset

Download and extract the weatherdb.zip to a directory for example `/home/user/weather_cityscapes`

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

## Regards

Don't hesitate to contact us if there are questions about the code or about the different options in the cfg file.
Thank you!!










