from models.deeplabv3_resnet101 import get_deeplabv3_resnet101
from models.deeplabv3_resnet50 import get_deeplabv3_resnet50
from models.fcn_resnet101 import get_fcn_resnet101
from models.fcn_resnet50 import get_fcn_resnet50


def get_model(name, **kwargs):
    models = {
        "fcn_resnet50": get_fcn_resnet50,
        "deeplabv3_resnet50": get_deeplabv3_resnet50,
        "fcn_resnet101": get_fcn_resnet101,
        "deeplabv3_resnet101": get_deeplabv3_resnet101,
        "unet_resnet50": get_fcn_resnet50,
    }
    
    if name not in models:
        raise ValueError(f"Model {name} não suportado. Opções: {list(models.keys())}")
    
    return models[name](**kwargs)