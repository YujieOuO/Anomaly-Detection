import random
import math
from torchvision import transforms
import torch 
import sys

#对dataloader里的数据，每次取出一个batch的数据，自定义格式输出
def cut_paste_collate_fn(batch):
    # cutPaste return 2 tuples of tuples we convert them into a list of tuples
    img_types = list(zip(*batch))
#     print(list(zip(*batch)))
    return [torch.stack(imgs) for imgs in img_types]
    

class CutPaste(object):
    """Base class for both cutpaste variants with common operations"""
    def __init__(self, colorJitter=0.1, transform=None):
        self.transform = transform
        
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = colorJitter,
                                                      contrast = colorJitter,
                                                      saturation = colorJitter,
                                                      hue = colorJitter)
    def __call__(self, org_img, img):
        # apply transforms to both images
        if self.transform:
            img = self.transform(img)
            org_img = self.transform(org_img)
        return org_img, img
    
class CutPasteNormal(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        area_ratio (list): list with 2 floats for maximum and minimum area to cut out
        aspect_ratio (float): minimum area ration. Ration is sampled between aspect_ratio and 1/aspect_ratio.
    """
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio

    def __call__(self, img):
        #TODO: we might want to use the pytorch implementation to calculate the patches from https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#RandomErasing
        h = img.size[0]
        w = img.size[1]
        
        # ratio between area_ratio[0] and area_ratio[1]
        ratio_area = random.uniform(self.area_ratio[0], self.area_ratio[1]) * w * h
        
        # sample in log space
        log_ratio = torch.log(torch.tensor((self.aspect_ratio, 1/self.aspect_ratio)))
        aspect = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()
        
        cut_w = int(round(math.sqrt(ratio_area * aspect)))
        cut_h = int(round(math.sqrt(ratio_area / aspect)))
        
        # one might also want to sample from other images. currently we only sample from the image itself
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)
        
        to_location_h = int(random.uniform(0, h - cut_h))
        to_location_w = int(random.uniform(0, w - cut_w))
        
        insert_box = [to_location_w, to_location_h, to_location_w + cut_w, to_location_h + cut_h]
        augmented = img.copy()
        augmented.paste(patch, insert_box)
        
        return super().__call__(img, augmented)
        
class CutpasteDisturb(CutPaste):
    """Random generation of three-channel color disturbance.
    Args:
        width_ratio(list):list with 2 floats for maximum and minimum width to add disturbance
        height_ratio(list):list with 2 floats for maximum and minimum height to add disturbance
    """
    def __init__(self,
                 width_ratio=[0.01, 0.05],
                 height_ratio=[0.01, 0.05],
                 color_value=[0, 255],
                 **kwags):
        super(CutpasteDisturb, self).__init__(**kwags)
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        self.color_value = color_value
        self.totensor = transforms.ToTensor()
        self.toPIL = transforms.ToPILImage()

    def __call__(self, img):
        H = img.size[0]
        W = img.size[1]

        # ratio between ratio[0] and ratio[1]
        r_height = random.uniform(*self.height_ratio)
        r_width = random.uniform(*self.width_ratio)

        # region of disturbance
        disturb_height = round(H * r_height)
        disturb_width = round(W * r_width)

        # beginning of the disturbance
        begin_disturb_height = random.randint(0, H - disturb_height)
        begin_disturb_width = random.randint(disturb_width, W - disturb_width)

        # three disturbance value
        value_1 = random.randint(*self.color_value)
        value_2 = random.randint(*self.color_value)
        value_3 = random.randint(*self.color_value)
        value_list = [value_1, value_2, value_3]

        # turn into tensor
        img_tensor = self.totensor(img)
        augmented = img_tensor.clone()

        # for RGB
        for channel in range(3):
            for height_step in range(disturb_height):
                # width offset
                width_offset = random.randint(0, disturb_width)
                for width_step in range(disturb_width):
                    # three channel color disturbance
                    augmented[channel][begin_disturb_height + height_step][
                        begin_disturb_width + width_step -
                        width_offset] = value_list[channel]

        # turn into image
        augmented = self.toPIL(augmented)

        return super().__call__(img, augmented)

class CutPasteScar(CutPaste):
    """Randomly copy one patche from the image and paste it somewere else.
    Args:
        width (list): width to sample from. List of [min, max]
        height (list): height to sample from. List of [min, max]
        rotation (list): rotation to sample from. List of [min, max]
    """
    def __init__(self, width=[2,16], height=[10,25], rotation=[-45,45], **kwags):
        super(CutPasteScar, self).__init__(**kwags)
        self.width = width
        self.height = height
        self.rotation = rotation
    
    def __call__(self, img):
        h = img.size[0]
        w = img.size[1]
        
        # cut region
        cut_w = random.uniform(*self.width)
        cut_h = random.uniform(*self.height)
        
        from_location_h = int(random.uniform(0, h - cut_h))
        from_location_w = int(random.uniform(0, w - cut_w))
        
        box = [from_location_w, from_location_h, from_location_w + cut_w, from_location_h + cut_h]
        patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*self.rotation)
        patch = patch.convert("RGBA").rotate(rot_deg,expand=True)
        
        #paste
        to_location_h = int(random.uniform(0, h - patch.size[0]))
        to_location_w = int(random.uniform(0, w - patch.size[1]))

        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_w, to_location_h), mask=mask)
        
        return super().__call__(img, augmented)
    
class CutPasteUnion(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.turb = CutpasteDisturb(**kwags)
    
    def __call__(self, img):
        r = random.uniform(0, 1)
        if r < 0.5:
            return self.normal(img)
        else:
            return self.turb(img)
            

class CutPaste3Way(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.scar = CutPasteScar(**kwags)
    
    def __call__(self, img):
        org, cutpaste_normal = self.normal(img)
        _, cutpaste_scar = self.scar(img)
        
        return org, cutpaste_normal, cutpaste_scar

