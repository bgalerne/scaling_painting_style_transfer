# Model download
Download the original VGG19 weights ```vgg_conv.pth``` from [here](https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8&export=download)
and save them in the ```model/```folder.

E.g. using gdown:
>```
>import gdown
>gdown.download("https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8", "model/vgg_conv.pth")
>```

**Credits:** Original weights from https://github.com/leongatys/PytorchNeuralStyleTransfer
