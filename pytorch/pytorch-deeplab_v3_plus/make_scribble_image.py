import json
from PIL import Image, ImageDraw, ImageOps
import os

label_dict = {"bg":(0,0,0), "tree":(1,1,1), "roof":(2,2,2), "grass":(3,3,3), "ground":(4,4,4), "obj":(5,5,5), "rail":(6,6,6)}
# label_dict = {"bg":(0,0,0), "tree":(1,1,1), "green_obj":(2,2,2), "yellow_grass":(3,3,3), "ground":(4,4,4), "obj":(5,5,5)}

bg_cnt = 0
tree_cnt = 0
roof_cnt = 0
grass_cnt = 0
ground_cnt = 0
obj_cnt = 0
rail_cnt=0

dir_path = "/home/qisens/2020.3~/rloss/data/tree_scribble/json"
scribble_path = "/home/qisens/2020.3~/rloss/data/tree_scribble/scribble_png"
for path, dirs, files in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(path, file)
        f = open(file_path)
        data = json.load(f)

        image_path = os.path.join(scribble_path, data["imagePath"])
        width, height = data["imageWidth"], data["imageHeight"]
        img = Image.new("RGB", (width, height), color=(255,255,255))
        draw = ImageDraw.Draw(img)

        for line in data["shapes"]:
            label = line["label"]
            points = line["points"]

            if label == "bg":
                bg_cnt += 1
            elif label == "tree":
                tree_cnt +=1
            elif label == "roof":
                roof_cnt += 1
            elif label == "grass":
                grass_cnt +=1
            elif label == "ground":
                ground_cnt += 1
            elif label == "rail":
                rail_cnt += 1
            else:
                obj_cnt += 1


            for idx, point in enumerate(points):
                if idx == 0:
                    first_pt = point
                else:
                    second_pt = point
                    draw.line((first_pt[0], first_pt[1], second_pt[0], second_pt[1]), fill=label_dict[label], width=5)
                    first_pt = second_pt

        gray_image = ImageOps.grayscale(img)
        gray_image.save(image_path)

print("bg: {}, tree: {}, roof:{}, grass:{}, ground:{} obj:{} rail:{}".format(bg_cnt, tree_cnt, roof_cnt, grass_cnt, ground_cnt, obj_cnt, rail_cnt))