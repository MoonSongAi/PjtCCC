from roboflow import Roboflow
import yaml
from PIL import Image

rf = Roboflow(api_key="API")
project = rf.workspace("sessac").project("project_object-igclo")
version = project.version(4)
dataset = version.download("yolov5")

file='C:/PjtCCC/Project_Object-4/data.yaml'

with open(file, 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print('▶ 원래 yaml 자료')
    print(data)

    data['train'] = 'C:/PjtCCC/Project_Object-4/train/images'
    data['test'] = 'C:/PjtCCC/Project_Object-4/test/images'
    data['val'] = 'C:/PjtCCC/Project_Object-4/valid/images'

    with open(file, 'w') as f:
        yaml.dump(data, f)

    print('▶ 수정된 yaml')
    print(data)

# python C:/PjtCCC/yolov5/train.py --data C:/PjtCCC/Project_Object-4/data.yaml --name runs/result --img 416 --batch 16 --epochs 100 --cfg C:/PjtCCC/yolov5/models/yolov5s.yaml --weights yolov5s.pt