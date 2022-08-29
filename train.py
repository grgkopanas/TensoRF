
import os
from tqdm.auto import tqdm
from opt import config_parser
import torch
import numpy as np

import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

import math
from torchinterp1d import Interp1d
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import imageio

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

def debugVisualizer(train_dataset, pdf, points):
    import polyscope as ps

    ps.init()

    ps_pc = ps.register_point_cloud("samples", points, radius=0.006)

    ps_x = ps.register_point_cloud("origin_x", np.array([[0., 0., 0.]]), enabled=True)
    ps_y = ps.register_point_cloud("origin_y", np.array([[0., 0., 0.]]), enabled=True)
    ps_z = ps.register_point_cloud("origin_z", np.array([[0., 0., 0.]]), enabled=True)
    ps_x.add_vector_quantity("dir_x", np.array([[1., 0., 0.]]), enabled=True, color=(1., 0., 0.))
    ps_y.add_vector_quantity("dir_y", np.array([[0., 1., 0.]]), enabled=True, color=(0., 1., 0.))
    ps_z.add_vector_quantity("dir_z", np.array([[0., 0., 1.]]), enabled=True, color=(0., 0., 1.))

    #ps_pc_diffuse.add_color_quantity("point_cloud_colors",
    #                                 self.diffuse_point_cloud.global_features[:, :3].detach().cpu().numpy(),
    #                                 enabled=True)

    cam_poses = train_dataset.world_o.cpu().detach().numpy()
    ps_cameras = ps.register_point_cloud("cameras", cam_poses, enabled=True)

    dirs = torch.einsum("ijk, j -> ik", train_dataset.poses[:,:3, :3].transpose(dim0=1, dim1=2), torch.tensor([0.0, 0.0, 1.0]))
    ps_cameras.add_vector_quantity("cam_dirs", np.array(dirs).squeeze(), enabled=True)
    ps_cameras.add_color_quantity("pdf", pdf[:, None].repeat(3, axis=1), enabled=True)

    ps.show()


class GKSampler:
    def __init__(self, train_dataset, batch_size):
        self.train_dataset = train_dataset
        self.rays = train_dataset.all_rays
        self.proj_mats = train_dataset.proj_mat
        self.world_o = train_dataset.world_o
        self.batch_size = batch_size


        N_CAMERAS = self.proj_mats.shape[0]

        points = torch.tensor([[0.0, 0.0, 0.0]])
        N_POINTS = points.shape[0]

        """
        # Random sample a point in 3D
        points = torch.rand(N_POINTS-1, 3)*16.0 - 8.0
        points = torch.cat((torch.tensor([[0.0, 0.0, 0.0]]), points), dim=0)
        N_POINTS = points.shape[0]

        # Homogenify
        hom_proj_mats = torch.cat((self.proj_mats, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(N_CAMERAS, 1, 1)), dim=1)
        hom_points = torch.cat((points, torch.tensor([[1.0]]).repeat(N_POINTS, 1)), dim=1)

        # Project Point to all cameras
        projected_points = torch.einsum('lkj, ij -> lik', hom_proj_mats, hom_points)
        positive_z_filter = projected_points[:, :, 2:3] > 0
        projected_points_divz = projected_points/projected_points[:, : , 2:3]
        incamera_filter = (projected_points_divz[:, :, 0:1] > 0) & (projected_points_divz[:, :, 0:1] < 800.0) & \
                          (projected_points_divz[:, :, 1:2] > 0) & (projected_points_divz[:, :, 1:2] < 800.0)
        """

        # Point-Camera direction
        PO_v = points.repeat(N_CAMERAS, 1, 1) - self.world_o.unsqueeze(1).repeat(1, N_POINTS, 1)
        #dirs = PO_v/PO_v.norm(dim=2, keepdim=True)

        #r, theta, phi
        sph_coord = self.appendSpherical_np(PO_v[:,0,:])[:, 3:]
        sph_coord[:, 1] = (sph_coord[:, 1] - math.pi/2.0)/(math.pi/2.0)  # pi/2.0 - pi to 0 - 1
        sph_coord[:, 2] = ((sph_coord[:, 2] + math.pi)/(2*math.pi))# -pi - pi to 0 - 1


        # 0 - 1 to 0 - H/W
        W = 600
        H = 300
        y = sph_coord[:,1]*H
        x = sph_coord[:,2]*W

        # Extend image in X axis to have a cyclic voronoi
        x_expand = np.concatenate((x, x-W, x+W), axis=0)
        y_expand = np.concatenate((y, y,   y), axis=0)

        # Grid to get the centers of all pixels
        grid = np.mgrid[0:H,0:W].transpose(1,2,0)
        polar_cameras = np.stack((y_expand, x_expand), axis=1) + 0.5

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(polar_cameras)
        _, indices = nbrs.kneighbors(grid.reshape(-1,2))
        indices = indices.reshape(H,W,1)%N_CAMERAS
        plt.imshow(((indices)%20), cmap="tab20")
        plt.scatter(x[np.array((self.world_o[:,0]>0) & (self.world_o[:,1]>0))],
                    y[np.array((self.world_o[:,0]>0) & (self.world_o[:,1]>0))], s=2.5, c="red")
        plt.scatter(x[np.array((self.world_o[:,0]<=0) | (self.world_o[:,1]<=0))],
                    y[np.array((self.world_o[:,0]<=0) | (self.world_o[:,1]<=0))], s=2.5, c="blue")
        plt.show()

        #cm = plt.cm.get_cmap('tab20')
        #cv2.imwrite('F:/output.png', cv2.cvtColor(255*cm((indices)%20).squeeze().astype("float32"), cv2.COLOR_RGB2BGR))
        D_theta = (math.pi/2.0) / H
        D_phi = 2 * math.pi / W
        perpixel_area = np.sin(((np.mgrid[:H] + 0.5) / H) * math.pi)[:, None].repeat(W, axis=1) * D_theta * D_phi
        per_cam_area = np.bincount(indices.reshape(H * W), weights=perpixel_area.reshape(H * W))

        self.cam_pdf = (per_cam_area / per_cam_area.sum())

        #debugVisualizer(self.train_dataset, per_cam_area, points)


    def appendSpherical_np(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)
        ptsnew[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
        return ptsnew

    def nextids(self):
        #p = np.zeros_like(self.cam_pdf)
        #p[:82] = 0.25 / p[:82].shape[0]
        #p[82:] = 0.75 / p[82:].shape[0]
        rand_images = np.random.choice(np.arange(self.rays.shape[0]), size=self.batch_size, p=self.cam_pdf)
        rand_pixels = torch.LongTensor(np.random.randint(low=0, high=self.rays.shape[1], size=self.batch_size))
        return rand_images, rand_pixels

    def nextids_noarea_onlydir(self):
        N_POINTS = 200
        N_CAMERAS = self.proj_mats.shape[0]
        # Random sample a point in 3D
        points = torch.rand(N_POINTS-1, 3)*16.0 - 8.0
        points = torch.cat((torch.tensor([[0.0, 0.0, 0.0]]), points), dim=0)

        # Homogenify
        hom_proj_mats = torch.cat((self.proj_mats, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(N_CAMERAS, 1, 1)), dim=1)
        hom_points = torch.cat((points, torch.tensor([[1.0]]).repeat(N_POINTS, 1)), dim=1)

        # Project Point to all cameras
        projected_points = torch.einsum('lkj, ij -> lik', hom_proj_mats, hom_points)
        positive_z_filter = projected_points[:, :, 2:3] > 0
        projected_points_divz = projected_points/projected_points[:, : , 2:3]
        incamera_filter = (projected_points_divz[:, :, 0:1] > 0) & (projected_points_divz[:, :, 0:1] < 800.0) & \
                          (projected_points_divz[:, :, 1:2] > 0) & (projected_points_divz[:, :, 1:2] < 800.0)

        # Point-Camera direction
        PO_v = points.repeat(N_CAMERAS, 1, 1) - self.world_o.unsqueeze(1).repeat(1, N_POINTS, 1)
        dirs = PO_v/PO_v.norm(dim=2, keepdim=True)

        # Construct Distance matrix 1.0 is close 0.0 is far away
        pairwise_inv_d = 3.14 - torch.acos(torch.einsum('kij,lij->ikl', dirs, dirs).clamp(-0.99999999, 0.99999999)).unsqueeze(-1)

        #TODO MISSING A FILTERING STEP OF THE CAMERAS THAT DONT ACTUALLY SEE THE POINT

        val, _ = torch.topk(pairwise_inv_d, k=5, dim=2)
        pdf_unormalized = 3.14 - val[:,:,1:,:].mean(dim=2)
        pdf = pdf_unormalized/pdf_unormalized.sum(dim=1, keepdim=True)

        # Sample the cameras inversely proportional to that.
        debugVisualizer(self.train_dataset, pdf, points)

        indexer = torch.LongTensor(np.random.randint(low=0, high=self.rays.shape[0], size=self.batch_size))
        return indexer

class SimpleSampler2:
    def __init__(self, rays, batch_size):
        self.rays = rays
        self.batch_size = batch_size

    def nextids(self):
        indexer = torch.LongTensor(np.random.randint(low=0, high=self.rays.shape[0], size=self.batch_size))
        return indexer

class SimpleSampler:
    def __init__(self, rays, batch_size):
        self.rays = rays
        self.batch_size = batch_size

    def nextids(self):
        rand_images = np.random.randint(0, self.rays.shape[0] - 1, size=self.batch_size)
        rand_pixels = torch.LongTensor(np.random.randint(low=0, high=self.rays.shape[1], size=self.batch_size))
        return rand_images, rand_pixels


        #self.curr += self.batch
        #if self.curr + self.batch > self.total:
        #    self.ids = torch.LongTensor(np.random.permutation(self.total))
        #    self.curr = 0
        #return self.ids[self.curr:self.curr+self.batch]

class PdfSampler:
    def __init__(self, batch_size, rays, rgbs, pdf):
        self.pdf = pdf
        self.rays = rays
        self.rgbs = rgbs
        self.batch_size = batch_size

    def inverse_transform_sampling(self, data, n_samples=1000000):

        data_im = data.squeeze()
        data_norm = data_im / data_im.sum()

        # CDF has one more value than PMF because we need to handle the bins
        cum_values = torch.zeros(data_norm.shape[0] + 1)
        cum_values[1:] = torch.cumsum(data_norm, dim=0)

        indexes = torch.arange(cum_values.shape[0]).float()
        # inv_cdf = interpolate.interp1d(cum_values, indexes.numpy())
        r = torch.rand(n_samples) * cum_values.max()
        samples = Interp1d()(cum_values, indexes, r).int().squeeze().clamp(max=data_im.shape[0] - 1)

        return samples

    def nextids(self):
        image = random.randint(0, self.rgbs.shape[0] - 1)
        n_rays = self.rays[image].shape[0]

        accepted_samples = torch.tensor([], dtype=torch.int)

        alpha = 0.5
        # Uniform
        sample_idxs = torch.randint(n_rays, (int(self.batch_size * alpha),), device='cpu')
        accepted_samples = torch.cat((accepted_samples, sample_idxs))

        # Pdf
        pdf_samples = self.inverse_transform_sampling(self.pdf[image], self.batch_size - accepted_samples.shape[0])
        accepted_samples = torch.cat((accepted_samples, torch.tensor(pdf_samples)))

        indexer = accepted_samples[:self.batch_size]
        indexer = indexer[torch.randperm(self.batch_size)]

        return image, indexer

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f"{args.ckpt[:-3]}.ply", bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)

    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)


    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    #allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    #if not args.ndc_ray:
    #    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)

    #trainingSampler = SimpleSampler2(train_dataset.all_rays, args.batch_size)
    #trainingSampler = PdfSampler(args.batch_size, train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_grads)
    trainingSampler = GKSampler(train_dataset, args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:


        rays_idx = trainingSampler.nextids()
        rays_train, rgb_train = train_dataset.all_rays[rays_idx],\
                                train_dataset.all_rgbs.reshape(train_dataset.all_rgbs.shape[0],
                                                               train_dataset.all_rgbs.shape[1]*train_dataset.all_rgbs.shape[2],
                                                               train_dataset.all_rgbs.shape[3])[rays_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)


        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)



        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            #if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
            #    allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
            #    trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

