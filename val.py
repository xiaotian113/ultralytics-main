import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics import YOLO
# 1. 加载模型
ckpt = torch.load('ultralytics-main/交付/Shape-IoU/best.pt', map_location='cpu')
if isinstance(ckpt, dict) and 'model' in ckpt:
    module = ckpt['model']
else:
    module = ckpt

for param in module.parameters():
    param.requires_grad = True

module = module.float()
def dummy_fuse(verbose=False):
    return module
module.fuse = dummy_fuse
module.eval()
module = module.cuda()
temp_model = YOLO('MYyolo11.yaml')
temp_model.model = module
temp_model.info()

# 2. 验证
args = {
    'data': 'ultralytics-main/SARDet-100K.yaml',
    'imgsz': 640,
    'batch': 16,
    'device': 'cuda',
    'workers': 8,
    'verbose': True,
    'plots': True,
    'save_json': False,
}
validator = DetectionValidator(args=args)
metrics = validator(model=module)

