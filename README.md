# 3D_scene_reconstruction_semantic
In this repository, I complete 3D Semantic Map for the HW2 of Perception and Decision Making in Intelligent Systems, NYCU, in fall 2022.

# Abstact
we will use two model to validate data collect from hw1 and reconstruct it, but in fact I have do some changing when collect semantic image. So just use data_collect.py to collect hw1_data.

# Example Result:
floor1 with apartment0:
floor1 with apartmentM:
3D scene of floor1 with apartment0:

# Quick Start
The requirement of the development environments:
- OS : ubuntu 18.04 , 20.04
- Python 3.6, 3.7 ( You can use conda to create new environment )
- opencv-contrib-python
- Open3d
- Habitat-lab and Habitat-sim<br>
following the link https://github.com/facebookresearch/habitat-lab
to until Installation step3.<br>
Note: using under code to install habitat_lab
```
/home/{user_name}/anaconda3/envs/habitat/bin/pip install -e .
```
- semantic-segmentation-pytorch<br>
download from the link https://github.com/CSAILVision/semantic-segmentation-pytorch
and select a model to train.<br>
All pretrained models can be found at: http://sceneparsing.csail.mit.edu/model/pytorch<br>
Note: I choose ade20k-resnet101-upernet as my pretrained model.
# Download Data
Download the apartment0 datasets from the link below.<br>
Apartment_0 : https://drive.google.com/file/d/1zHA2AYRtJOmlRaHNuXOvC_OaVxHe56M4/view?usp=sharing<br>
Note : You can change agent_state.position to set the agent in
the first or second floor ( (0, 0, 0) for first floor and (0, 1, 0) for
second floor.

Download the apartmentM datasets from the link below.<br>
Note : I only use apartment_1, apartment_2, frl_apartment_0, frl_apartment_1<br>
apartment_1 : https://drive.google.com/file/d/1xrBvrYvSI8vmAX5OiTtY1nDrm9bCpUpy/view?usp=sharing<br>
apartment_2 : https://drive.google.com/file/d/13wYbpZXR4YSLpxPSo2-O8cv0pYp1DFoE/view?usp=sharing<br>
frl_apartment_0 : https://drive.google.com/file/d/1_n5QL9nmwEz5zO3kuZvufoC23s_sluf1/view?usp=sharing<br>
frl_apartment_1 : https://drive.google.com/file/d/1WfOAvTF0H6fVxpMXLNbCBk7xqCcV9e0q/view?usp=sharing<br>
room_0 : https://drive.google.com/file/d/16jI_peN1fJrRKDp38YzzM-x5GzZNstuS/view?usp=sharing<br>
room_1 : https://drive.google.com/file/d/1y04o5kG-7uJ8JLw5lXGS20m_-sEb1Epv/view?usp=sharing<br>
room_2 : https://drive.google.com/file/d/1Xmqbw0RabdpnUxTtsMTWf9-h7CPGf9Yg/view?usp=sharing<br>
hotel_0 : https://drive.google.com/file/d/1Lv0QgkyLnAefS0RsYEV5KUJyw4C5Cwmu/view?usp=sharing<br>
office_0 : https://drive.google.com/file/d/1-xRcBM5eBu-OZ0bZa1Dmt8OnHIIo1UyF/view?usp=sharing<br>
office_1 : https://drive.google.com/file/d/1aNeYQ2W8d1g-qsOtdXUtzxpfkTKfILu7/view?usp=sharing<br>

# Data Collection
auto collect data from dataset_folder, ex: env_apartment0, env_apartmentM, which has download data from above link.<br>
Note : need to change self.\_scenes in data_generator.py to which scenes you want to collect in dataset_folder
```
python data_generator.py --dataset_folder [dataset_folder] --output [output folder] 
```
run the following command and use WAD to move camera and auto save data in hw1_data <br>
rgb image, depth image, ground truth trajectory will auto save when you move.<br>
F key will finish the program.<br>
O key will clean all rgb image, depth image, ground truth trajectory saved before.<br>
Note : You can change agent_state.position to set the agent in
the first or second floor, (0, 0, 0) for first floor and (0, 1, 0) for
second floor.
```
conda activate habitat
cd hw2
python data_collect.py
```
use following command to get odgt files for each folder we collect data, ex: img_apartment0, img_apartmentM, hw1_data<br>
Note : need to change some variables in the file to decide which data folder to get odgt file.
```
python img_to_odgt.py
```

# Train Model
need change in yaml :<br>
1. root_dataset means where train dataset's root dataset is.
2. num_class : 101 (because hw1_data only use 101 label of semantic)，then go to model.py revise part in build_decoder, only load weight have same shape in pretrained model.<br>
3. start_epoch : 50 (because pretrained model's ckpt is 50)<br>
4. num_epoch actullay is end epoch, num_epoch - start_epoch equal to how many epoch you want to train.<br>
5. val's check_point : "epoch_60.pth"

using the following command to train model.<br>  
Note : \-\-gpus 0 means using cuda:0
```
cd semantic-segmentation-pytorch
python train.py --config config/ade20k-resnet101-upernet.yaml --gpus 0
```
# Structure of directory
```
habitat-lab
  ......
hw2
  +- env_apartment0
    +- apartment_0
  +- env_apartmentM
    +- apartment_1
    +- apartment_2
    +- frl_apartment_0
    +- frl_apartment_1
  +- img_apartment0
    +- annotations
      +- test
      +- train
      +- val
    +- images 
      ......
    +- depth
      ......
    +- metal_training.odgt
    +- metal_validation.odgt
  +- img_apartmentM
    ......
  +- hw1_data
    +- floor1
      +- annotations
        +- semantic_0.png
        .....
      +- depth
        +- depth_0.png
        ......
      +- images
        +- rgb_0.png
        ......
      +- result_smantic_apartment0
        +- semantic_0.png
        ......
      +- result_smantic_apartmentM
        ......
      +- hw1_floor1.odgt
    +- floor2
      ......
  +- semantic-segmentation-pytorch
  +- data_collect.py
  +- data_generator.py
  +- img_to_odgt.py
  +-  3d_semantic_map.py
  
```
