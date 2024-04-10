import torch

def relative_error_ninf(input_tensor, target_tensor):
  with torch.no_grad():
    return torch.max(torch.abs(input_tensor-target_tensor))/torch.max(torch.abs(target_tensor)).item()

def relative_error_l2(input_tensor, target_tensor):
  with torch.no_grad():
    return torch.norm(input_tensor-target_tensor, p=2)/torch.norm(target_tensor, p=2).item()

def print_relative_error(input_tensor, target_tensor):
  print("Relative errors: Ninf = %1.2e ; L2 = %1.2e"%(relative_error_ninf(input_tensor, target_tensor),
                                                  relative_error_l2(input_tensor, target_tensor)))

def resize_height_pil(PIL_img, new_height):
  if PIL_img.height==new_height:
    return(PIL_img.copy())
  else:
    ratio = float(PIL_img.width)/float(PIL_img.height)
    new_width = round(ratio*float(new_height))
    new_size = (new_width , new_height)
    return(PIL_img.resize(new_size))

def zoomout_pil(PIL_img, invfactor):
  # invfactor divides the size of the image
  if invfactor==1:
    return(PIL_img.copy())
  else:
    new_size = (round(float(x)/float(invfactor)) for x in PIL_img.size)
    return(PIL_img.resize(new_size))
  
  
def crop_width_multiple_pil(PIL_img, multiple):
  new_width = multiple*(PIL_img.width//multiple)
  return(PIL_img.crop((0,0,new_width,PIL_img.height)))

def crop_multiple_pil(PIL_img, multiple):
  # crop the extremeties of an image so that each dimension is a multiple of "multiple"
  new_width = multiple*(PIL_img.width//multiple)
  new_heigth = multiple*(PIL_img.height//multiple)
  return(PIL_img.crop((0,0,new_width,new_heigth)))







