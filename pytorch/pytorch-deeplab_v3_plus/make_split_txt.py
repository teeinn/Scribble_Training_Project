import os
import shutil
from random import shuffle

#-----------------------------------------------------------------------------if val / train dataset are not splited
dir_path = "/media/qisens/2tb1/python_projects/Scribble_Training_Project/data/tree_scribble/scribble_png"
path, dirs, files = next(os.walk(dir_path))
shuffle(files)

#strain/val --  8:2
file_cnt = len(files)
train_cnt = round(file_cnt * 0.8)

train_files = files[:train_cnt]
val_files = files[train_cnt:]

train_txt = "/media/qisens/2tb1/python_projects/Scribble_Training_Project/data/tree_scribble/split_data/train.txt"
val_txt = "/media/qisens/2tb1/python_projects/Scribble_Training_Project/data/tree_scribble/split_data/val.txt"

with open(train_txt, "w") as file1:
    for idx, f in enumerate(train_files):
        name, ext = os.path.splitext(f)
        file1.write(name+"\n")
file1.close()
with open(val_txt, "w") as file2:
    for f in val_files:
        name, ext = os.path.splitext(f)
        file2.write(name+"\n")
file2.close()




#-------------------------------------------------------------------------------------if val / train dataset are already splited
# dir_path = "/media/qisens/2tb1/goodroof_solarpanel_parkinglot_rooftop_facility__output_with_augmented"
# scribble_img_path = os.path.join(dir_path, "train/scribble_img_new")
# original_img_path = os.path.join(dir_path, "train/JPEGImages")
# txt_path = os.path.join(dir_path, "split_data/train.txt")
#
# # path, dirs, files = next(os.walk(scribble_img_path))
# # with open(txt_path, "w") as file1:
# #     for idx, f in enumerate(files):
# #         name, ext = os.path.splitext(f)
# #         file1.write(name+"\n")
# # file1.close()
#
# with open(txt_path, "w") as file1:
#     for path, dirs, files in os.walk(scribble_img_path):
#         for file in files:
#             file_origin_path = os.path.join(path, file)
#             if '.xml.png' in file:
#                 new_file = file.replace('.xml.png', '.png')
#                 file_new_path = os.path.join(path, new_file)
#                 os.rename(file_origin_path, file_new_path)
#                 name, ext = os.path.splitext(new_file)
#             else:
#                 name, ext = os.path.splitext(file)
#             file1.write(name+"\n")
# file1.close()