from math import ceil
import torch


# Convention for multi-blocks description

# We partitionate the original image following a square grid.

# Regargind image size and partition:
# * Subimages share an overlap area of overlap_margin_size = 256 pixels in original size.
# * For each block one has:
#  * The coordinates of the copied part of the original image.
#  * The coordinate length of the overlap on each side.
#  * The coordinates of the area corresponding to the strict partition (coordinates of copied part - overlap area, can be deduced using the two first info).
#  * One needs to know the size of the VGG layers of the full image (even though we never compute them fully at once).
# See comments in the __init__ function below.



class VGG_BY_BLOCKS:
  def __init__(self, vgg, img_torch, style_layers, inner_block_size = 512, content_layers=[], overlap_margin_size = 256, verbose_mode=True):
    self.img_torch = img_torch

    # Block section:
    self.inner_block_size = inner_block_size
    self.ov = overlap_margin_size
    if(self.ov>inner_block_size):
      print("warning: Inner block size is too small for optimal results")
      while(self.ov>inner_block_size):
        self.ov = self.ov//2
        print("self.ov is: %d"%(self.ov))
    _, nc, h, w = img_torch.shape
    # check number of channels:
    if(nc!=3):
      print("WARNING: only for RGB images")
    self.nblocks_h = int(ceil(h/self.inner_block_size))
    self.nblocks_w = int(ceil(w/self.inner_block_size))
    #print("VGG_BY_BLOCKS made of %d x %d (First guess)" % (self.nblocks_h, self.nblocks_w))

    # if last block does not have full margin, remove it:
    if (h < self.inner_block_size*self.nblocks_h + self.ov and self.nblocks_h>1):
      self.nblocks_h -= 1
    if (w < self.inner_block_size*self.nblocks_w + self.ov and self.nblocks_w>1):
      self.nblocks_w -= 1
    ## if last block has no margin, remove it:
    #if (h < self.inner_block_size*self.nblocks_h and self.nblocks_h>1):
    #  self.nblocks_h -= 1
    #if (w < self.inner_block_size*self.nblocks_w and self.nblocks_w>1):
    #  self.nblocks_w -= 1
    ## if before last block does not have full margin, remove last block:
    #if (h < self.inner_block_size*(self.nblocks_h-1)+self.ov and self.nblocks_h>1):
    #  self.nblocks_h -= 1
    #if (w < self.inner_block_size*(self.nblocks_w-1)+self.ov and self.nblocks_w>1):
    #  self.nblocks_w -= 1

    # Need an odd number of blocks to avoid border issues (blocks with zero padding)???


    self.nblocks = self.nblocks_h*self.nblocks_w
    if verbose_mode:
        print("VGG_BY_BLOCKS made of %d x %d = %d blocks" % (self.nblocks_h, self.nblocks_w, self.nblocks))
    # list of coordinates for each block:
    self.full_block_original_coord_list = [] # coordinates of block with margin
    self.inner_block_original_coord_list = [] # coordinates of innner blocks: all inner blocks form a partition of the image domain
    self.block_overlap_list = [] # margin used for each block (either self.ov or 0 in each direction)
    for i in range(self.nblocks_h):
      top_ov = (0 if i==0 else self.ov)
      bot_ov = (0 if i==self.nblocks_h-1 else self.ov)
      for j in range(self.nblocks_w):
        left_ov = (0 if j==0 else self.ov)
        right_ov = (0 if j==self.nblocks_w-1 else self.ov)
        inner_coord_tuple = (i*self.inner_block_size,
                                h if i==self.nblocks_h-1 else min((i+1)*self.inner_block_size,h),
                                j*self.inner_block_size,
                                w if j==self.nblocks_w-1 else min((j+1)*self.inner_block_size,w))
        tuple_ov = (top_ov, bot_ov, left_ov, right_ov)
        full_coord_tuple = (inner_coord_tuple[0]-top_ov,
                            inner_coord_tuple[1]+bot_ov,
                            inner_coord_tuple[2]-left_ov,
                            inner_coord_tuple[3]+right_ov)
        self.full_block_original_coord_list.append(full_coord_tuple)
        self.inner_block_original_coord_list.append(inner_coord_tuple)
        self.block_overlap_list.append(tuple_ov)

    # VGG section: list of used layers, list of sizes of (virtual) VGG layers etc.
    self.vgg = vgg
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.style_scale_list = self.vgg_layer_scale(style_layers)
    self.content_scale_list = self.vgg_layer_scale(content_layers)
    self.style_nchannel_list = self.vgg_layer_nchannel(self.style_layers)
    self.content_nchannel_list = self.vgg_layer_nchannel(self.content_layers)
    self.style_layers_spatial_size_list = []
    for idx, layer in enumerate(self.style_layers):
      self.style_layers_spatial_size_list.append((h//self.style_scale_list[idx],
                                        w//self.style_scale_list[idx]))
    self.content_layers_spatial_size_list = []
    for idx, layer in enumerate(self.content_layers):
      self.content_layers_spatial_size_list.append((h//self.content_scale_list[idx],
                                        w//self.content_scale_list[idx]))


  def vgg_layer_nchannel(self, vgg_layer_list):
    vgg_nchannel_dict = {
        "r11": 64,
        "r12": 64,
        'p1': 64,
        "r21": 128,
        "r22": 128,
        'p2': 128,
        "r31": 256,
        "r32": 256,
        "r33": 256,
        "r34": 256,
        'p3': 256,
        "r41": 512,
        "r42": 512,
        "r43": 512,
        "r44": 512,
        'p4': 512,
        "r51": 512,
        "r52": 512,
        "r53": 512,
        "r54": 512,
        'p5': 512}
    vgg_nchannel_list = [vgg_nchannel_dict[layer] for layer in vgg_layer_list]
    return(vgg_nchannel_list)

  def vgg_layer_scale(self,vgg_layer_list):
    vgg_scale_dict = {
        "r11": 1,
        "r12": 1,
        'p1': 2,
        "r21": 2,
        "r22": 2,
        'p2': 4,
        "r31": 4,
        "r32": 4,
        "r33": 4,
        "r34": 4,
        'p3': 8,
        "r41": 8,
        "r42": 8,
        "r43": 8,
        "r44": 8,
        'p4': 16,
        "r51": 16,
        "r52": 16,
        "r53": 16,
        "r54": 16,
        'p5': 32}
    vgg_scale_list = [vgg_scale_dict[layer] for layer in vgg_layer_list]
    return(vgg_scale_list)

  #method to compute a full VGG layer by blocks:
  def compute_content_layer_by_blocks(self):
    if len(self.content_layers) != 1:
      print("Currently only supported for a single content layer")
    content_layer_features = torch.zeros((1,
                                         self.content_nchannel_list[0],
                                         self.content_layers_spatial_size_list[0][0],
                                         self.content_layers_spatial_size_list[0][1]),
                                         device=self.img_torch.device)
    for block_idx in range(self.nblocks):
      #print()
      #print("block_idx:", block_idx)
      block_torch = self.extract_block_with_ov(block_idx)
      #print("block_torch.shape : ", block_torch.shape)
      with torch.no_grad():
        out = self.vgg(block_torch, self.content_layers)
        vgg_out_layer = out[0]
      b,c,h,w = vgg_out_layer.shape
      inner_coord_tuple = self.inner_block_original_coord_list[block_idx]
      #print("inner_coord_tuple: ", *inner_coord_tuple)
      inner_coord_tuple_layer = tuple((x//self.content_scale_list[0] for x in inner_coord_tuple))
      #print("inner_coord_tuple_layer: ", *inner_coord_tuple_layer)
      tuple_ov = self.block_overlap_list[block_idx]
      #print("tuple_ov: ", *tuple_ov)
      tuple_ov_layer = (x//self.content_scale_list[0] for x in tuple_ov)
      top_ov,bot_ov,left_ov,right_ov = tuple_ov_layer
      #print("tuple_ov_layer: ", top_ov,bot_ov,left_ov,right_ov)
      vgg_out_layer_crop = vgg_out_layer[:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
      #print("vgg_out_layer_crop.shape : ", vgg_out_layer_crop.shape)
      content_layer_features[:,:,
                          inner_coord_tuple_layer[0]:inner_coord_tuple_layer[1],
                          inner_coord_tuple_layer[2]:inner_coord_tuple_layer[3]] = vgg_out_layer_crop.clone()
    return([content_layer_features])

  def extract_block_with_ov(self, block_idx):
    full_coord_tuple = self.full_block_original_coord_list[block_idx]
    block_torch = self.img_torch[:,:,full_coord_tuple[0]:full_coord_tuple[1],
                                 full_coord_tuple[2]:full_coord_tuple[3]]
    return(block_torch)


  def global_Gram_matrices(self):
    # initialize Gram matrices with zeros:
    with torch.no_grad():
      Gram_matrices = [torch.zeros((1,ncl,ncl),
                                  device=self.img_torch.device
                                  ) for ncl in self.style_nchannel_list]
      for block_idx in range(self.nblocks):
        full_coord_tuple = self.full_block_original_coord_list[block_idx]
        block_torch = self.extract_block_with_ov(block_idx)
        out = self.vgg(block_torch, self.style_layers)
        tuple_ov = self.block_overlap_list[block_idx]
        for nl, G in enumerate(Gram_matrices):
          top_ov,bot_ov,left_ov,right_ov = (x//self.style_scale_list[nl] for x in tuple_ov)
          b,c,h,w = out[nl].shape
          vgg_out_layer_crop = out[nl][:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
          # update Gram matrix (just sum product of feature over spatial coordinates):
          F = vgg_out_layer_crop.reshape(b, c, -1)
          G.add_(torch.bmm(F, F.transpose(1,2)))

      # normalize by full layer size:
      for nl, G in enumerate(Gram_matrices):
        G.div_(self.style_layers_spatial_size_list[nl][0] *
              self.style_layers_spatial_size_list[nl][1])
    return(Gram_matrices)

  def global_Gram_matrices_means_and_stds(self):
    # initialize Gram matrices with zeros:
    with torch.no_grad():
      Gram_matrices = [torch.zeros((1,ncl,ncl),
                                  device=self.img_torch.device
                                  ) for ncl in self.style_nchannel_list]
      layers_means = [torch.zeros((1,ncl,1,1),
                                  device=self.img_torch.device
                                  ) for ncl in self.style_nchannel_list]

      for block_idx in range(self.nblocks):
        full_coord_tuple = self.full_block_original_coord_list[block_idx]
        block_torch = self.extract_block_with_ov(block_idx)
        out = self.vgg(block_torch, self.style_layers)
        tuple_ov = self.block_overlap_list[block_idx]
        for nl, (G, mu) in enumerate(zip(Gram_matrices, layers_means)):
          top_ov,bot_ov,left_ov,right_ov = (x//self.style_scale_list[nl] for x in tuple_ov)
          b,c,h,w = out[nl].shape
          vgg_out_layer_crop = out[nl][:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
          # update layers means (just sum over spatial coordinates):
          mu.add_(torch.sum(vgg_out_layer_crop, (2,3), keepdim=True))
          # update Gram matrix (just sum product of feature over spatial coordinates):
          F = vgg_out_layer_crop.reshape(b, c, -1)
          G.add_(torch.bmm(F, F.transpose(1,2)))

      # normalize by full layer size:
      for nl, G in enumerate(Gram_matrices):
        G.div_(self.style_layers_spatial_size_list[nl][0] *
              self.style_layers_spatial_size_list[nl][1])
      for nl, mu in enumerate(layers_means):
        mu.div_(self.style_layers_spatial_size_list[nl][0] *
              self.style_layers_spatial_size_list[nl][1])
      # compute standard deviations
      layers_stds = [torch.sqrt(torch.diag(G.squeeze()).view(1,-1,1,1) - mu**2)
                    for G, mu in zip(Gram_matrices, layers_means)]
    return(Gram_matrices, layers_means, layers_stds)

  def Gram_loss_layer_grad_outputs(self, style_weights, vgg_out, Gram_differences):
    loss_layer_grad_outputs = []
    for idx, vgg_layer in enumerate(vgg_out):
      b,c,h,w = vgg_layer.shape
      # full layer size
      hf = self.style_layers_spatial_size_list[idx][0]
      wf = self.style_layers_spatial_size_list[idx][1]
      vgg_layer_pixel_list = vgg_layer.reshape(b, c, h*w)
      vgg_layer_mixed = torch.bmm(vgg_layer_pixel_list.transpose(1,2), Gram_differences[idx])
      vgg_layer_mixed = vgg_layer_mixed.transpose(1,2)
      vgg_layer_mixed = vgg_layer_mixed.reshape(b, c, h, w).div(hf*wf)
      vgg_layer_mixed = vgg_layer_mixed.mul(style_weights[idx] * 4./Gram_differences[idx].numel())
      loss_layer_grad_outputs.append(vgg_layer_mixed)
    return(loss_layer_grad_outputs)


  def global_Gram_loss_with_gradient(self, style_targets, style_weights):
    # style_targets is a list of Gram matrices
    # Step 0: Initialize gradient of the image:
    if self.img_torch.grad is None:
      self.img_torch.grad = torch.zeros(self.img_torch.shape,
                                        device = self.img_torch.device)
    # Step 1: Compute differences betweem current Gram matrices and targets
    # and loss value
    Gram_matrices = self.global_Gram_matrices()
    diff_Gram_matrices = [G - style_targets[nl]
                          for nl, G in enumerate(Gram_matrices)]
    full_loss = torch.tensor(0.,device = self.img_torch.device)
    for idx, G in enumerate(diff_Gram_matrices):
      full_loss.add_(torch.sum(G**2).mul_(style_weights[idx]/G.numel()))

    # Step 2: Compute Gram loss gradient using backprop for each block:

    for block_idx in range(self.nblocks):
      # extract block
      block_torch = self.extract_block_with_ov(block_idx)
      # compute vgg layers of blocks
      vgg_out = self.vgg(block_torch, self.style_layers)
      # compute gradient with respect to block
      loss_layer_grad_outputs = self.Gram_loss_layer_grad_outputs(style_weights, vgg_out[:len(self.style_layers)], diff_Gram_matrices)
      # autograd for block:
      block_torch_grad = torch.autograd.grad(vgg_out, block_torch, loss_layer_grad_outputs)[0]
      # add inner part of this gradient to the full gradient
      inner_coord_tuple = self.inner_block_original_coord_list[block_idx]
      tuple_ov = self.block_overlap_list[block_idx]
      self.img_torch.grad[:,:,
                          inner_coord_tuple[0]:inner_coord_tuple[1],
                          inner_coord_tuple[2]:inner_coord_tuple[3]].add_(
              block_torch_grad[:,:,tuple_ov[0]:block_torch.shape[2]-tuple_ov[1],
                               tuple_ov[2]:block_torch.shape[3]-tuple_ov[3]])
    return(full_loss)

  def global_content_plus_Gram_loss_with_gradient(self, style_targets, style_weights,
                                                  content_targets, content_weights):
    # style_targets is a list of Gram matrices
    # content_targets is a list of a single torch tensor
    # the function is an enrichment of global_Gram_loss_with_gradient
    # Step 0: Initialize gradient of the image:
    if self.img_torch.grad is None:
      self.img_torch.grad = torch.zeros(self.img_torch.shape,
                                        device = self.img_torch.device)
    # Step 1: Compute differences betweem current Gram matrices and targets
    # and loss value
    Gram_matrices = self.global_Gram_matrices()
    diff_Gram_matrices = [G - style_targets[nl] for nl, G in enumerate(Gram_matrices)]
    full_loss = torch.tensor(0.,device = self.img_torch.device)
    for idx, G in enumerate(diff_Gram_matrices):
      full_loss.add_(torch.sum(G**2).mul_(style_weights[idx]/G.numel()))
    #print("loss at step 1: ",full_loss.item())
    # Step 2: Compute content+Gram loss gradient using backprop for each block: Add content loss parts to full_loss
    for block_idx in range(self.nblocks):
      # extract block
      block_torch = self.extract_block_with_ov(block_idx)
      # compute vgg layers of blocks
      vgg_out = self.vgg(block_torch, self.style_layers+self.content_layers)
      # compute gradient with respect to block
      # Gram loss:
      loss_layer_grad_outputs = self.Gram_loss_layer_grad_outputs(style_weights,
                                                                  vgg_out[:len(self.style_layers)],
                                                                  diff_Gram_matrices)
      # content loss: extract block from content target and compute the difference

      full_coord_tuple = self.full_block_original_coord_list[block_idx]
      full_coord_tuple_layer = tuple((x//self.content_scale_list[0] for x in full_coord_tuple))
      target_content_block = content_targets[0][:,:,
                                                full_coord_tuple_layer[0]:full_coord_tuple_layer[1],
                                                full_coord_tuple_layer[2]:full_coord_tuple_layer[3]]
      content_layer_grad_outputs = content_weights[0]*2/torch.numel(content_targets[0])*(vgg_out[len(self.style_layers)]-target_content_block)
      loss_layer_grad_outputs.append(content_layer_grad_outputs)
      # autograd for block:
      block_torch_grad = torch.autograd.grad(vgg_out, block_torch, loss_layer_grad_outputs)[0]
      # add inner part of this gradient to the full gradient
      inner_coord_tuple = self.inner_block_original_coord_list[block_idx]
      tuple_ov = self.block_overlap_list[block_idx]
      self.img_torch.grad[:,:,
                          inner_coord_tuple[0]:inner_coord_tuple[1],
                          inner_coord_tuple[2]:inner_coord_tuple[3]].add_(
              block_torch_grad[:,:,tuple_ov[0]:block_torch.shape[2]-tuple_ov[1],
                               tuple_ov[2]:block_torch.shape[3]-tuple_ov[3]])
      # add block contribution to content loss: mse of inner block:
      b,c,h,w = vgg_out[len(self.style_layers)].shape
      tuple_ov = self.block_overlap_list[block_idx]
      tuple_ov_layer = (x//self.content_scale_list[0] for x in tuple_ov)
      top_ov,bot_ov,left_ov,right_ov = tuple_ov_layer
      vgg_out_layer_crop = vgg_out[len(self.style_layers)][:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
      target_content_block_crop = target_content_block[:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
      full_loss.add_(torch.sum((vgg_out_layer_crop - target_content_block_crop)**2
                              ).mul_(content_weights[0]/torch.numel(content_targets[0])))
    #print("loss at step 2: ",full_loss.item())
    return(full_loss)

  def Gram_means_stds_loss_layer_grad_outputs(self,
                                              style_weights,
                                              vgg_out,
                                              Gram_differences,
                                              layers_means,
                                              diff_layers_means,
                                              factors_stds
                                             ):
    loss_layer_grad_outputs = []
    weights_Gram_matrices, weights_layers_means, weights_layers_stds = style_weights
    for idx, vgg_layer in enumerate(vgg_out):
      b,c,h,w = vgg_layer.shape
      # full layer size
      hf = self.style_layers_spatial_size_list[idx][0]
      wf = self.style_layers_spatial_size_list[idx][1]
      # Gram component:
      vgg_layer_pixel_list = vgg_layer.reshape(b, c, h*w)
      vgg_layer_mixed = torch.bmm(vgg_layer_pixel_list.transpose(1,2), Gram_differences[idx])
      vgg_layer_mixed = vgg_layer_mixed.transpose(1,2)
      vgg_layer_mixed = vgg_layer_mixed.reshape(b, c, h, w)
      Gram_factor = weights_Gram_matrices[idx]*4./(Gram_differences[idx].numel()*hf*wf)
      vgg_layer_mixed = vgg_layer_mixed.mul_(Gram_factor)
        # need to divide by numel() because of MSE_loss that divides by numel()
      # Add mean component:
      mean_factor = weights_layers_means[idx]*2./(diff_layers_means[idx].numel()*hf*wf)
      vgg_layer_mixed += mean_factor*diff_layers_means[idx]
      # Add std component:
      std_factor = weights_layers_stds[idx]*2./(factors_stds[idx].numel()*hf*wf)
      vgg_layer_mixed += std_factor * ((vgg_layer-layers_means[idx])*factors_stds[idx])
      # append to list
      loss_layer_grad_outputs.append(vgg_layer_mixed)
    return(loss_layer_grad_outputs)

  ####
  # This is the main method used in spst.py
  ####
  def global_content_plus_Gram_means_stds_loss_with_gradient(self,
                                                             style_targets, style_weights,
                                                             content_targets, content_weights):
    # style_targets is a tuple of the form :
    #  (list of Gram matrices, list of layer means, list of layer stds)
    # such that the output of global_Gram_matrices_means_and_stds
    # style_weights is a list of the form:
    #  (list of weights for Gram loss, list of weights for mean loss, list of weights for std loss)
    # content_targets is a list of a single torch tensor
    # content_weights is a list of a single weight
    # the function is an enrichment of global_Gram_loss_with_gradient
    # Step 0: Initialize gradient of the image:
    if self.img_torch.grad is None:
      self.img_torch.grad = torch.zeros(self.img_torch.shape,
                                        device = self.img_torch.device)
    # Step 1: Compute differences betweem statistics and style loss value
    Gram_matrices, layers_means, layers_stds = self.global_Gram_matrices_means_and_stds()
    Gram_matrices_targets, layers_means_targets, layers_stds_targets = style_targets
    # Gram matrices:
    diff_Gram_matrices = [G - G_target
                          for G, G_target in zip(Gram_matrices, Gram_matrices_targets)]
    # Layers means:
    diff_layers_means = [mu-mu_target
                         for mu, mu_target in zip(layers_means, layers_means_targets)]
    # Layers stds:
    factors_stds = [(s-s_target)/s
                   for s, s_target in zip(layers_stds, layers_stds_targets)]

    # compute loss:
    full_loss = torch.tensor(0.,device = self.img_torch.device)
    weights_Gram_matrices, weights_layers_means, weights_layers_stds = style_weights
    for G, w in zip(diff_Gram_matrices, weights_Gram_matrices):
      full_loss.add_(torch.sum(G**2).mul_(w/G.numel()))
    for mu, w in zip(diff_layers_means, weights_layers_means):
      full_loss.add_(torch.sum(mu**2).mul_(w/mu.numel()))
    for s, s_target, w in zip(layers_stds, layers_stds_targets, weights_layers_stds):
      full_loss.add_(torch.sum((s-s_target)**2).mul_(w/s.numel()))

    #print("loss at step 1: ",full_loss.item())
    # Step 2: Compute content+(Gram, means and stds) loss gradient using backprop for each block:
    # Also add content loss parts to full_loss
    for block_idx in range(self.nblocks):
      # extract block
      block_torch = self.extract_block_with_ov(block_idx)
      # compute vgg layers of blocks
      vgg_out = self.vgg(block_torch, self.style_layers+self.content_layers)
      # compute gradient with respect to block
      # style loss:
      loss_layer_grad_outputs = self.Gram_means_stds_loss_layer_grad_outputs(
                                              style_weights,
                                              vgg_out[:len(self.style_layers)],
                                              diff_Gram_matrices,
                                              layers_means,
                                              diff_layers_means,
                                              factors_stds)
      # content loss: extract block from content target and compute the difference
      full_coord_tuple = self.full_block_original_coord_list[block_idx]
      full_coord_tuple_layer = tuple((x//self.content_scale_list[0] for x in full_coord_tuple))
      target_content_block = content_targets[0][:,:,
                                                full_coord_tuple_layer[0]:full_coord_tuple_layer[1],
                                                full_coord_tuple_layer[2]:full_coord_tuple_layer[3]]
      content_layer_grad_outputs = content_weights[0]*2/torch.numel(content_targets[0])*(vgg_out[len(self.style_layers)]-target_content_block)
      loss_layer_grad_outputs.append(content_layer_grad_outputs)
      # autograd for block:
      block_torch_grad = torch.autograd.grad(vgg_out, block_torch, loss_layer_grad_outputs)[0]
      # add inner part of this gradient to the full gradient
      inner_coord_tuple = self.inner_block_original_coord_list[block_idx]
      tuple_ov = self.block_overlap_list[block_idx]
      self.img_torch.grad[:,:,
                          inner_coord_tuple[0]:inner_coord_tuple[1],
                          inner_coord_tuple[2]:inner_coord_tuple[3]].add_(
              block_torch_grad[:,:,tuple_ov[0]:block_torch.shape[2]-tuple_ov[1],
                               tuple_ov[2]:block_torch.shape[3]-tuple_ov[3]])
      # add block contribution to content loss: mse of inner block:
      b,c,h,w = vgg_out[len(self.style_layers)].shape
      tuple_ov = self.block_overlap_list[block_idx]
      tuple_ov_layer = (x//self.content_scale_list[0] for x in tuple_ov)
      top_ov,bot_ov,left_ov,right_ov = tuple_ov_layer
      vgg_out_layer_crop = vgg_out[len(self.style_layers)][:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
      target_content_block_crop = target_content_block[:,:,top_ov:(h-bot_ov),left_ov:(w-right_ov)]
      full_loss.add_(torch.sum((vgg_out_layer_crop - target_content_block_crop)**2
                              ).mul_(content_weights[0]/torch.numel(content_targets[0])))
    #print("loss at step 2: ",full_loss.item())
    return(full_loss)
