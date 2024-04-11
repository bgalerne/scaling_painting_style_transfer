# Scaling Painting Style Transfer
Official repository for the paper "Scaling Painting Style Transfer" by 
[Bruno Galerne](https://www.idpoisson.fr/galerne/), 
[Lara Raad](http://dev.ipol.im/~lraad/), 
[José Lezama](https://iie.fing.edu.uy/~jlezama/),
and
[Jean-Michel Morel](https://sites.google.com/site/jeanmichelmorelcmlaenscachan)

![Teaser for SPST](img/teaser.png)


## Requirements
* PyTorch 2.1
* PILLOW

## Model download
Download the original VGG19 weights ```vgg_conv.pth``` from [here](https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8&export=download)
and save them in the ```model/```folder.

E.g. using gdown:
>```
>import gdown
>gdown.download("https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8", "model/vgg_conv.pth")
>```

**Credits:** Original weights from https://github.com/leongatys/PytorchNeuralStyleTransfer


## Usage:

The main module is ```spst.py```.

```
python spst.py -c path_to_content_image -s path_to_style_image
```

>**Example:**
>```
>python spst.py -c content_images/treilles_01_z2.png -s path_to_style_image -s style_images/van_Gogh_wheat_field_with_cypresses.jpg
>```

Output results are written in the ```results/```folder.

**Full option list:**
```
 python spst.py [-h] [-c CONTENT] [-s STYLE]
               [--results_root_dir RESULTS_ROOT_DIR]
               [--iterations_mode {spst-baseline,spst-fast,debug}]
               [--device DEVICE]
               [--rescale_mode {downscale_style,upscale_content,none}]
               [--minimal_scale_threshold MINIMAL_SCALE_THRESHOLD]
               [--wmeanstd WMEANSTD] [--maxgpufit MAXGPUFIT]
  -h, --help            show this help message and exit
  -c CONTENT, --content CONTENT
                        path to content image
  -s STYLE, --style STYLE
                        path to style image
  --results_root_dir RESULTS_ROOT_DIR
                        path to root directory where results directory is
                        created
  --iterations_mode {spst-baseline,spst-fast,debug}
                        choice for the number of L-BFGS iterations at each
                        scale: spst-baseline: long but best results, spst-
                        fast: fewer iterations on large scales, debug: only 10
                        iterations at each scale to check installation/memory
                        usage
  --device DEVICE, -d DEVICE
                        device number 0,1,... for cuda:0, cuda:1
  --rescale_mode {downscale_style,upscale_content,none}
                        rescale_mode = 'downscale_style' (default): downscale
                        style image so that min sidelength of both image is
                        the same; rescale_mode = 'upscale_content': upscale
                        content image so that max sidelength of both images is
                        the same; escale_mode = 'none': no rescale
  --minimal_scale_threshold MINIMAL_SCALE_THRESHOLD
                        size threshold to stop dividing by 2 when fixing the
                        number of scales. 1023 (default): minimal scale is
                        between 512 and 1023; 2047: minimal scale is between
                        1024 and 2047; 4095: minimal scale is between 2048 and
                        4095
  --wmeanstd WMEANSTD   weight for additional "mean and std" style loss term
                        (set to 0 for Gram loss only)
  --maxgpufit MAXGPUFIT
                        maximal size in Megapixel for doing the opitmization
                        in the GPU (optimization computation is switched to
                        CPU if larger). Default value 64 (ie 8000x8000) is for
                        a GPU with 40 GB VRAM, to adapt according to your
                        device.
```

## TODO:
* Code for texture synthesis
* Description of source code files




