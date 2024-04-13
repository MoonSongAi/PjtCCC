from roboflow import Roboflow
import yaml
from PIL import Image

rf = Roboflow(api_key="s1ydUhtNRaBVlGnU4lr0")
project = rf.workspace("sessac").project("blackboard-bold_project")
version = project.version(3)
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