# SpotRust

This is the repository for the code from the paper 'Deep Learning Corrosion Detection with Confidence'.  
The base network is derived from [HRNetV2](https://github.com/HRNet/HRNet-Semantic-Segmentation) (provided for reference you do not need the original HRNetV2 for this to run)  

Code written by: Will Nash

## Environment
We recommend that you use a python environment and package manager, [Anaconda](https://anaconda.org) or [miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then create the environment from terminal as follows:  
    ```
     $ conda create --name <env> --file SpotRust_packages.txt  
    ```

## Usage
   ### Model Training
1. Download the pretrained ImageNet weights from the [HRNetV2 repository](https://github.com/HRNet/HRNet-Semantic-Segmentation).
1. Generate a text file with the links to your dataset images and masks using tab separated values. Note that we are unable to provide the dataset used for the paper due to restrictions from the industry partner.
1. Configure the hyperparameter file for training, an example is provided in [corrosion_MCDO.json](/corrosion_MCDO.json).
1. Run the training script as follows (we recommend to prefix nohup and append & to run in headless mode):
    ```
    $ python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 train.py corrosion_MCDO.json --pretrained ../SpotRust/hrnet_cocostuff_3617_torch04.pth > $(date +%Y_%m_%d).txt
    ```
1. Models will be saved to `saved/[model_arch]/`

    ### Inference
1. Select an image to run inference on, use var_infer.py for variational (HRNet_Var) or Monte-Carlo dropout (HRNet_do) models, and ensemble_infer.py for ensemble models, e.g.:  
    ```
   $ python var_infer.py --model 'saved/HRNet_do/21-09-07[09.23]' --n_MC 24 --thresh 0.8 --image '../DATA/training/images/IMG_2876.JPG' --gt '../DATA/training/gt/IMG_2876_gt.jpg' 
   ```
    
    or...  

    ```
    $ python ensemble_infer.py --models saved/HRNet/21-06-17\[09.51\]/ saved/HRNet/21-07-04\[13.36\]/ saved/HRNet/21-07-05\[04.03\]/ --thresh 0.8 --image '../DATA/training/images/IMG_2876.JPG' --gt '../DATA/training/gt/IMG_2876_gt.jpg' 
    ```
1. Output images will be saved to `figures/[model_arch]`


