# This is the main module for
# Scaling Painting Style Transfer
# Authors: Bruno Galerne, Lara Raad, José Lezama, Jean-Michel Morel
# https://arxiv.org/abs/2212.13459
# Project page:
# https://www.idpoisson.fr/galerne/scaling_painting_style_transfer/index.html
#
################################################################################
# Parse arguments:
import argparse
parser = argparse.ArgumentParser(description="Scaling Painting Style Transfer: High-quality style transfer for ultra high-resolution images")

parser.add_argument('-c', '--content', type=str, help='path to content image', default = 'content_images/treilles_01.png')
parser.add_argument('-s', '--style', type=str, help='path to style image', default = 'style_images/van_Gogh_wheat_field_with_cypresses.jpg')
parser.add_argument('--results_root_dir', type=str, default='results', help='path to root directory where results directory is created')
parser.add_argument('--iterations_mode', choices=['spst-baseline', 'spst-fast', 'debug'], default='spst-fast', help='choice for the number of L-BFGS iterations at each scale: spst-baseline: long but best results, spst-fast: fewer iterations on large scales, debug: only 10 iterations at each scale to check installation/memory usage')
parser.add_argument('--device','-d', type=str, default='0', help='device number 0,1,... for cuda:0, cuda:1')
parser.add_argument('--rescale_mode', choices=['downscale_style', 'upscale_content', 'none'], default='downscale_style',
    help = "rescale_mode = 'downscale_style' (default): downscale style image so that min sidelength of both image is the same; rescale_mode = 'upscale_content': upscale content image so that max sidelength of both images is the same; escale_mode = 'none': no rescale")
parser.add_argument('--minimal_scale_threshold', default = 1023, type=int, help='size threshold to stop dividing by 2 when fixing the number of scales. 1023 (default): minimal scale is between 512 and 1023; 2047: minimal scale is between 1024 and 2047; 4095: minimal scale is between 2048 and 4095')
parser.add_argument('--wmeanstd', default = 1000, type=int, help = 'weight for additional "mean and std" style loss term (set to 0 for Gram loss only)')
parser.add_argument('--maxgpufit', default = 70, type=int, help = 'maximal size in Megapixel for doing the opitmization in the GPU (optimization computation is switched to CPU if larger). Default value 64 (ie 8000x8000) is for a GPU with 40 GB VRAM, to adapt according to your device.')

args = parser.parse_args()

iterations_mode = args.iterations_mode
content_img_name = args.content
style_img_name = args.style
rescale_mode = args.rescale_mode
# used in: while image_sizes[0] > minimal_scale_threshold divide by 2
minimal_scale_threshold = args.minimal_scale_threshold
wmeanstd = args.wmeanstd
results_root_dir = args.results_root_dir
maxgpufit = args.maxgpufit

################################################################################

import time
import os
from datetime import datetime

import torch
from torch import optim

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
# because of OSError: image file is truncated
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:"+args.device if torch.cuda.is_available() else "cpu")
print("Device is :", device)
print(torch.__version__)

# import functions from auxiliary files:
from vgg_by_blocks import VGG_BY_BLOCKS  # main import with contributions
from vgg_gatys import VGG, prep, postp   # code from Gatys' repo https://github.com/leongatys/PytorchNeuralStyleTransfer
from utils import resize_height_pil, zoomout_pil


################################################################################
#get VGG network (from https://github.com/leongatys/PytorchNeuralStyleTransfer)
vgg = VGG()
model_dir = os.path.join(os.getcwd(), 'model')
#Original Gatys et al. VGG weights should be downloaded here into model/ folder:
#Available here:
#https://drive.google.com/uc?id=1lLSi8BXd_9EtudRbIwxvmTQ3Ms-Qh6C8&export=download
vgg.load_state_dict(torch.load(os.path.join(model_dir,'vgg_conv.pth')))


for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.to(device)


start = datetime.now()

#define layers, loss functions, weights and compute optimization targets
style_layers = ['r11','r21','r31','r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers

#these are good weights settings:
weights_Gram_matrices = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [1e0]
# weights for block functions:
weights_layers_means = [w*n**2*wmeanstd/100 for n, w in zip([64,128,256,512,512], weights_Gram_matrices)]
weights_layers_stds = weights_layers_means
style_weights = (weights_Gram_matrices, weights_layers_means, weights_layers_stds)

#load input images:
# preprocessing: Get size for multi-scale
# content image:
content_img_pil_hr = Image.open(content_img_name).convert("RGB")
print("Original content image has size: ", content_img_pil_hr.size[1],' x ', content_img_pil_hr.size[0])
wco, hco = content_img_pil_hr.size

# style image:
style_img_pil_hr = Image.open(style_img_name).convert("RGB")
print("Original style image has size: ", style_img_pil_hr.size[1],' x ', style_img_pil_hr.size[0])

###########################################
# rescale image section
###########################################

rescale_flagdict = {'content': '.png',
                         'style': '.png'}

if rescale_mode=='downscale_style':
    if min(style_img_pil_hr.width, style_img_pil_hr.height) > 1.1*min(hco,wco):
      print("rescale_mode is ", rescale_mode)
      print("Target style image has larger resolution than the content, downscaling style image")
      ratio = float(min(hco,wco))/float(min(style_img_pil_hr.width, style_img_pil_hr.height))
      new_height = round(float(style_img_pil_hr.height)*ratio)
      style_img_pil_hr = resize_height_pil(style_img_pil_hr, new_height)
      rescale_flagdict['style'] = '_rescaled_'+str(new_height)+'.png'
if rescale_mode=='upscale_content':
    if min(style_img_pil_hr.width, style_img_pil_hr.height) > 1.1*min(hco,wco):
      print("rescale_mode is ", rescale_mode)
      print("Target style image has larger resolution than the content, upscaling content image")
      ratio = float(float(min(style_img_pil_hr.width, style_img_pil_hr.height)/min(hco,wco)))
      new_height = round(float(hco)*ratio)
      content_img_pil_hr = resize_height_pil(content_img_pil_hr, new_height)
      wco, hco = content_img_pil_hr.size
      rescale_flagdict['content'] = '_rescaled_'+str(new_height)+'.png'

# determine number of scales
wco, hco = content_img_pil_hr.size
image_sizes = [min(wco,hco)]
while image_sizes[0]>minimal_scale_threshold:
  image_sizes.insert(0,image_sizes[0]//2)
nbscales = len(image_sizes)

# Print sizes:
print("Used content image has size: ", content_img_pil_hr.size[1],' x ', content_img_pil_hr.size[0])
print("Used style image has size: ", style_img_pil_hr.size[1],' x ', style_img_pil_hr.size[0])
print("Number of scales: ", nbscales)


# define optimization parameters for each scale:
if iterations_mode == 'spst-baseline':
    iters = [300 for idx in range(nbscales)]; iters[0] = 600 # SPST baseline
elif iterations_mode == 'spst-fast':
    iters = [max(600//(3**s),30) for s in range(nbscales)] # SPST-fast
elif iterations_mode == 'debug':
    iters = [10 for idx in range(nbscales)] #  for debug

history_sizes = [10 for idx in range(nbscales)]; history_sizes[0] = 100

# create folder for results:
content_base_string = os.path.splitext(os.path.basename(content_img_name))[0]
style_base_string = os.path.splitext(os.path.basename(style_img_name))[0]
now = datetime.now()
dt_string=now.strftime("%Y%m%d_%H%M%S")
result_dir=os.path.join(results_root_dir, dt_string+'_'+content_base_string+'_'+style_base_string)
os.makedirs(result_dir, exist_ok=True)
print("Created folder: ", result_dir)
# save script for reproducibility:
os.system('cp %s "%s"' % (__file__, result_dir+'/'))
# write both input images:
content_img_pil_hr.save(os.path.join(result_dir, content_base_string+rescale_flagdict['content']))
style_img_pil_hr.save(os.path.join(result_dir, style_base_string+rescale_flagdict['style']))

for idx in range(nbscales):

  print('Scale '+str(idx+1)+'/'+str(nbscales))
  torch.cuda.synchronize()
  scale_start = datetime.now()

  # load input images:  content and style images
  # content image:
  # downgrade resolution to current scale:
  content_img_pil = zoomout_pil(content_img_pil_hr, 2**(nbscales-1-idx))
  content_image = prep(content_img_pil).unsqueeze(0)
  content_image = content_image.to(device)
  # if the image is too large, the optimizer is moved to the GPU.
  if content_img_pil.height*content_img_pil.width <= maxgpufit*1e6:
    opt_img_dev = device
  else:
    opt_img_dev = 'cpu'
  print("Optimization on:", opt_img_dev)

  # style image:
  # downgrade resolution to current scale:
  style_img_pil = zoomout_pil(style_img_pil_hr, 2**(nbscales-1-idx))
  style_image = prep(style_img_pil).unsqueeze(0)
  style_image = style_image.to(device)

  # opt_img: initialize with content for first scale, else zoomin using pil.
  if idx == 0:
    opt_img = content_image.clone().to(opt_img_dev)
  else:
    # write intermediate result on disc and upscale the current opt_img using pil
    with torch.no_grad():
      opt_img_pil = postp(opt_img.cpu().squeeze())
      result_name = 'result_sc_'+str(idx)+'_of_'+str(nbscales)+'--iters'
      for it in iters:
        result_name +='_'+str(it)
      result_name += '_cw_'+str(content_weights[0])+'_wmstd_'+str(wmeanstd)+'.png'
      # write intermediate result
      opt_img_pil.save(os.path.join(result_dir,result_name))
      # resize opt_img_pil to current size of content_img_pil (x2 zoomin up to even/odd sizes)
      opt_img_pil = opt_img_pil.resize(content_img_pil.size)
      opt_img = prep(opt_img_pil).unsqueeze(0).to(opt_img_dev)


  # compute style and content VGG targets with blocks:
  vgg_blocks_style_img = VGG_BY_BLOCKS(vgg, style_image, style_layers, content_layers = [])
  style_targets_by_blocks = vgg_blocks_style_img.global_Gram_matrices_means_and_stds()
  vgg_blocks_content_img = VGG_BY_BLOCKS(vgg, content_image, style_layers, content_layers = content_layers)
  content_targets_by_blocks = vgg_blocks_content_img.compute_content_layer_by_blocks()
  del style_image # free GPU memory
  del content_image
  #torch.cuda.empty_cache()

  opt_img.requires_grad = True
  # print("opt_img.requires_grad", opt_img.requires_grad,
  #   "opt_img.device", opt_img.device)
  # print("opt_img_device.requires_grad", opt_img_device.requires_grad,
  # "opt_img_device.device", opt_img_device.device)
  optimizer = optim.LBFGS([opt_img], history_size=history_sizes[idx])
  n_iter = [0]
  max_iter = iters[idx]
  show_iter = max_iter//10
  show = True

  while n_iter[0] <= max_iter:
    def closure():
      optimizer.zero_grad()
      if opt_img_dev == 'cpu':
          # (not the usual case, optimization steps on the cpu)
          # copy opt_img on device, compute gradient, and move gradient to cpu:
          opt_img_device = opt_img.clone().detach().to(device)
          opt_img_device.requires_grad = True
          vgg_blocks_opt_img = VGG_BY_BLOCKS(vgg, opt_img_device, style_layers,
                                           content_layers = content_layers,
                                           verbose_mode=False)
          hand_loss = vgg_blocks_opt_img.global_content_plus_Gram_means_stds_loss_with_gradient(
                                                        style_targets_by_blocks, style_weights,
                                                        content_targets_by_blocks, content_weights
            )
          opt_img.grad = opt_img_device.grad.to(opt_img.device)
      else:
          # optimization steps on the same GPU as the VGG network
          vgg_blocks_opt_img = VGG_BY_BLOCKS(vgg, opt_img, style_layers,
                                           content_layers = content_layers,
                                           verbose_mode=False)
          hand_loss = vgg_blocks_opt_img.global_content_plus_Gram_means_stds_loss_with_gradient(
                                                        style_targets_by_blocks, style_weights,
                                                        content_targets_by_blocks, content_weights)

      #print loss
      if show and (n_iter[0]==0 or n_iter[0]%show_iter == (show_iter-1)):
        #print('Iteration: %d, loss: %1.2e'%(n_iter[0]+1, loss.item()))
        print('Iteration: %d, hand_loss: %1.2e'%(n_iter[0]+1, hand_loss.item()))
        #print("opt_img.requires_grad", opt_img.requires_grad,
        #    "opt_img.device", opt_img.device)
        #print("opt_img_device.requires_grad", opt_img_device.requires_grad,
        #    "opt_img_device.device", opt_img_device.device)
      n_iter[0]+=1
      return hand_loss

    optimizer.step(closure)
  torch.cuda.synchronize()
  c = datetime.now() - scale_start
  # returns (minutes, seconds)
  minutes = divmod(c.total_seconds(), 60)
  print('Time for scale : '+str(idx+1)+'/'+str(nbscales), int(minutes[0]), 'minutes',
                                 int(minutes[1]), 'seconds')

# write final result in result folder:
with torch.no_grad():
    opt_img_pil = postp(opt_img.cpu().squeeze())
result_name = 'result_sc_'+str(idx+1)+'_of_'+str(nbscales)+'--iters'
for it in iters:
  result_name +='_'+str(it)
result_name += '_cw_'+str(content_weights[0])+'_wmstd_'+str(wmeanstd)+'.png'
opt_img_pil.save(os.path.join(result_dir,result_name))

torch.cuda.synchronize()
del optimizer # free memory
c = datetime.now() - start
minutes = divmod(c.total_seconds(), 60)     ## returns (minutes, seconds)
print('Total running time: ', int(minutes[0]), 'minutes',
                                int(minutes[1]), 'seconds')

# Write companion .info file
f = open(os.path.join(result_dir,result_name.replace(".png",".info")), "a")
minutes_float = c.total_seconds() / 60.
f.write('time in min.: '+"{:.2f}".format(minutes_float)+'\n')
f.write('nbscales: '+str(nbscales)+'\n')
f.write('iters: '+str(iters)+'\n')
f.write('content_img_name: '+str(content_img_name)+'\n')
f.write('style_img_name: '+str(style_img_name)+'\n')
f.write('Result folder: '+str(result_dir)+'\n')
f.write('rescale_mode: '+str(rescale_mode)+'\n')
f.close()
