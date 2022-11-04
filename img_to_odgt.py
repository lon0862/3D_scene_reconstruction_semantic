import os
import cv2
import json
from tqdm import tqdm

def odgt(img_path):
    seg_path = img_path.replace('images','annotations')
    seg_path = seg_path.replace('.jpg','.png')
    # extra for hw1_data
    seg_path = seg_path.replace('rgb','semantic')
    
    if os.path.exists(seg_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        odgt_dic = {}
        # img_path = img_path.replace('img_apartment0/','')
        # seg_path = seg_path.replace('img_apartment0/','')
        img_path = img_path.replace('hw1_data/'+floor+'/','')
        seg_path = seg_path.replace('hw1_data/'+floor+'/','')
        odgt_dic["fpath_img"] = img_path
        odgt_dic["fpath_segm"] = seg_path
        odgt_dic["width"] = h
        odgt_dic["height"] = w
        return odgt_dic
    else:
        # print('the corresponded annotation does not exist')
        # print(img_path)
        return None


if __name__ == "__main__":
    floor = 'floor2'
    modes = [floor]
    # modes = ['train','val']
    saves = ['hw1_'+floor+'.odgt'] # ['metal_training.odgt', 'metal_validation.odgt']

    for i, mode in enumerate(modes):
        save = saves[i]
        # dir_path = f"img_apartment0/images/{mode}"
        dir_path = f"hw1_data/{mode}/images"
        img_list = os.listdir(dir_path)
        img_list.sort()
        img_list = [os.path.join(dir_path, img) for img in img_list]

        # with open(f'img_apartment0/{save}', mode='wt', encoding='utf-8') as myodgt:
        with open(f'hw1_data/{mode}/{save}', mode='wt', encoding='utf-8') as myodgt:
            for i, img in enumerate(tqdm(img_list)):
                a_odgt = odgt(img)
                if a_odgt is not None:
                    myodgt.write(f'{json.dumps(a_odgt)}\n')
