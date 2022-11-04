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
- Habitat-lab and Habitat-sim
following the link https://github.com/facebookresearch/habitat-lab
to until Installation step3.<br>
note: using under code to install habitat_lab
```
/home/{user_name}/anaconda3/envs/habitat/bin/pip install -e .
```
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
