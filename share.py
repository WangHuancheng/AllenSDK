from pycococreatortools import pycococreatortools
import cv2
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def new_func():
    coco_output = {
        "info":{},
        "licenses": [],
        "categories": [{
            'id': 1,
            'name': 'your_name',
            'supercategory': 'your_sc',
    }],
        "images": [],
        "annotations": []
    }
    category_info = {'id': 1, 'is_crowd': False}
    ID = 0
    category_info = {'id': 1, 'is_crowd': False}
ID = 0
for i in range(500):
    image = cv2.imread('/image{}.png'.format(i))
    h,w,c=image.shape
    image_info=pycococreatortools.create_image_info(i,'image_{}.png'.format(i),[w,h])
    coco_output['images'].append(image_info)
    for j in range(len(mask[i])):
        annotation_info = pycococreatortools.create_annotation_info(ID,ophys_experiment_id,category_info,mask[i][j])
        ID+=1
        coco_output["annotations"].append(annotation_info)
with open("/train.json", "w") as outfile:
    json.dump(coco_output,outfile,cls=NpEncoder)
new_func()