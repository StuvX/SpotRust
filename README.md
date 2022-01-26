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

#### Pretrained Models

The pretrained models are provided at the following Google Drive links: 

* [HRNet [4.35 GB]](https://drive.google.com/file/d/196yj1ZpuuSn1Uhb8LmKANV0hmnPc2o3F/view?usp=sharing)
* [HRNet_do [487 MB]](https://drive.google.com/file/d/12d6je9A8YOvz_9To3R0MgJzaMUu1UrRZ/view?usp=sharing)
* [HRNet_var [521 MB]](https://drive.google.com/file/d/11GymBbJeyHkq1Td_ThSmGi4AAto20A5z/view?usp=sharing)

Download and extract the files into the 'saved'  directory, note that the model referenced in the hypes file is relative to the script path, you may need to edit it if your directory structure differs from the original. 

#### Model Training

To train on your own dataset follow these steps:

1. Download the pretrained ImageNet weights from the [HRNetV2 repository](https://github.com/HRNet/HRNet-Semantic-Segmentation).
1. Generate a text file with the links to your dataset images and masks using tab separated values. Note that we are unable to provide the dataset used for the paper due to restrictions from the industry partner.
1. Configure the hyperparameter file for training, an example is provided in [corrosion_MCDO.json](/corrosion_MCDO.json).
1. Run the training script as follows (we recommend to prefix nohup and append & to run in headless mode):
    ```
    $ python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 train.py corrosion_MCDO.json --pretrained ../SpotRust/hrnet_cocostuff_3617_torch04.pth > $(date +%Y_%m_%d).txt
    ```
1. Models will be saved to `saved/[model_arch]/`

 #### Inference  
    
6. Select an image to run inference on, use var_infer.py for variational (HRNet_Var) or Monte-Carlo dropout (HRNet_do) models, and ensemble_infer.py for ensemble models, e.g.:  
    ```
    $ python var_infer.py --model 'saved/HRNet_bayes_all/21-12-28[17.09]' --n_MC 24 --out_res 512 512 --thresh 0.75 --image '../DATA/training/images/IMG_2876.JPG' --gt '../DATA/training/gt/IMG_2876_gt.jpg'; 
    ```
    or...  
    
    ```
    $ python ensemble_infer.py --models 'saved/HRNet/21-12-12[14.49]/' --thresh 0.75 --out_res 512 512 --image '../DATA/training/images/IMG_2876.JPG' --gt '../DATA/training/gt/IMG_2876_gt.jpg'; 
    ```
1. Output images will be saved to `figures/[model_arch]`


----
Copyright 2022 Will Nash

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
