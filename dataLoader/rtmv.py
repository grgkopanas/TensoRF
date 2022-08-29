import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
from torch import nn
from pathlib import Path
import OpenEXR
import Imath

from .ray_utils import *

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, imgs):
        imgs = imgs.mean(dim=3).unsqueeze(1)
        x = self.filter(imgs)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

def get_exr_rgb(path):
    I = OpenEXR.InputFile(path)
    dw = I.header()['displayWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    data = [np.fromstring(c, np.float32).reshape(size) for c in I.channels('RGB', Imath.PixelType(Imath.PixelType.FLOAT))]
    img = np.dstack(data)
    #img = img**2.2
    img = np.clip(5.0*img, 0, 1)
    # convert colour to sRGB
    #img = np.where(img<=0.0031308, 12.92*img, 1.055*np.power(img, 1/2.4) - 0.055)
    return img

class RTMVDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [0.1, 3.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth
    
    def read_meta(self):

        if self.split=="train":
            self.ids = list(range(135))
        else:
            self.ids = list(range(135, 150))

        with open(os.path.join(self.root_dir, f"{self.ids[0]:05d}.json"), 'r') as f:
            meta = json.load(f)

        self.img_wh = (int(meta["camera_data"]["width"]), int(meta["camera_data"]["height"]))
        w, h = self.img_wh

        self.focal = meta["camera_data"]["intrinsics"]["fx"]

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_grads = []
        self.all_masks = []
        self.all_depth = []

        #idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        json_cameras = [os.path.join(self.root_dir,f) for f in os.listdir(self.root_dir) if f.endswith(".json")]
        for id in tqdm(self.ids, desc=f'Loading data {self.split} ({len(self.ids)})'):#img_list:#
            json_cam = os.path.join(self.root_dir, f"{id:05d}.json")
            with open(json_cam, 'r') as f:
                frame = json.load(f)

            #pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            pose = np.array(frame["camera_data"]["cam2world"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{Path(json_cam).stem}.exr")
            self.image_paths += [image_path]
            img = get_exr_rgb(image_path)
            img = self.transform(img)  # (4, h, w)

            img = img.unsqueeze(0).permute(0, 2, 3, 1)
            grad = Sobel()(img)

            img = img.view(-1, 3)
            grad = grad.view(-1, 1)

            self.all_rgbs += [img]
            self.all_grads += [grad]


            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_grads = torch.stack(self.all_grads, 0)

#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample
