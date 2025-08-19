from models.resnet import resnet18, resnet50, resnet101
from models.googlenet import GoogleNet
from models.VGG import vgg19
from models.EfficientNet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from models.ViT import vit_base_patch16_224
def GetModel(par,num_classes=None):
    ncls = par.num_classes if num_classes is None else num_classes
    if par.model == 'ResNet18':
        model = resnet18(num_classes=ncls)
    elif par.model == 'ResNet50':
        model = resnet50(num_classes=ncls
                         )
    elif par.model == 'ResNet101':
        model = resnet101(num_classes=ncls)
    elif par.model == 'EfficientNet_b0':
        model = efficientnet_b0(num_classes=ncls)
    elif par.model == 'EfficientNet_b3':
        model = efficientnet_b3(num_classes=ncls)
    elif par.model == 'ViT':
        model = vit_base_patch16_224(par,num_classes=ncls)

    elif par.model == 'VGG19':
        model = vgg19(num_classes=ncls)
    elif par.model == 'GoogleNet':
        model = GoogleNet(num_classes=ncls)
    return model
