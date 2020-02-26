from torchsummary import summary
import torchvision.models as models

m = list()
# m.append(("rensnet18", models.resnet18()))
# m.append(("alexnet", models.alexnet()))
# m.append(("vgg16", models.vgg16()))
m.append(("squeezenet1-0", models.squeezenet1_0()))
m.append(("squeezenet1-1", models.squeezenet1_1()))
# m.append(("densenet16", models.densenet161()))
# m.append(("inceptionv3", models.inception_v3()))
# m.append(("googlenet", models.googlenet()))
# m.append(("shuffletnetv2", models.shufflenet_v2_x1_0()))
# m.append(("mobilenetv2", models.mobilenet_v2()))
# m.append(("resnext50", models.resnext50_32x4d()))
# m.append(("wide_resnet50", models.wide_resnet50_2()))
# m.append(("mnasnet1", models.mnasnet1_0()))

for name, mod in m:
    print(name)
    summary(mod, (3, 64, 64))