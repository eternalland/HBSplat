# Shree KRISHNAya Namaha
# Differentiable warper implemented in PyTorch. Warping is done on batches.
# Tested on PyTorch 1.8.1
# Author: Nagabhushan S N
# Last Modified: 27/09/2021

from typing import Tuple, Optional

import torch
import torch.nn.functional as F

import torch

class Warper:
    def __init__(self, resolution: tuple = None, device: str = 'gpu0'):
        self.resolution = resolution
        self.device = self.get_device(device)
        return

    def forward_warp(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                     transformation1: torch.Tensor, transformation2: torch.Tensor, intrinsic1: torch.Tensor,
                     intrinsic2: Optional[torch.Tensor]) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
        bilinear splatting.
        All arrays should be torch tensors with batch dimension and channel first
        :param frame1: (b, 3, h, w). If frame1 is not in the range [-1, 1], either set is_image=False when calling
                        bilinear_splatting on frame within this function, or modify clipping in bilinear_splatting()
                        method accordingly.
        :param mask1: (b, 1, h, w) - 1 for known, 0 for unknown. Optional
        :param depth1: (b, 1, h, w)
        :param transformation1: (b, 4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
        :param transformation2: (b, 4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
        :param intrinsic1: (b, 3, 3) camera intrinsic matrix
        :param intrinsic2: (b, 3, 3) camera intrinsic matrix. Optional
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()

        assert frame1.shape == (b, 3, h, w)
        assert mask1.shape == (b, 1, h, w)
        assert depth1.shape == (b, 1, h, w)
        assert transformation1.shape == (b, 4, 4)
        assert transformation2.shape == (b, 4, 4)
        assert intrinsic1.shape == (b, 3, 3)
        assert intrinsic2.shape == (b, 3, 3)

        frame1 = frame1.to(self.device)
        mask1 = mask1.to(self.device)
        depth1 = depth1.to(self.device)
        transformation1 = transformation1.to(self.device)
        transformation2 = transformation2.to(self.device)
        intrinsic1 = intrinsic1.to(self.device)
        intrinsic2 = intrinsic2.to(self.device)

        trans_points1 = self.compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                        intrinsic2)
        trans_coordinates = trans_points1[:, :, :, :2, 0] / trans_points1[:, :, :, 2:3, 0]
        trans_depth1 = trans_points1[:, :, :, 2, 0].unsqueeze(1)

        grid = self.create_grid(b, h, w).to(trans_coordinates)
        flow12 = trans_coordinates.permute(0, 3, 1, 2) - grid

        warped_frame2, mask2 = self.bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
        warped_depth2 = self.bilinear_splatting(trans_depth1, mask1, trans_depth1, flow12, None,
                                                is_image=False)[0]
        return warped_frame2, mask2, warped_depth2, flow12

    def compute_transformed_points(self, depth1: torch.Tensor, transformation1: torch.Tensor, transformation2: torch.Tensor,
                                   intrinsic1: torch.Tensor, intrinsic2: Optional[torch.Tensor]):
        """
        Computes transformed position for each pixel location
        """
        if self.resolution is not None:
            assert depth1.shape[2:4] == self.resolution
        b, _, h, w = depth1.shape
        if intrinsic2 is None:
            intrinsic2 = intrinsic1.clone()
        # c12
        transformation = torch.bmm(transformation2, torch.linalg.inv(transformation1))  # (b, 4, 4)

        x1d = torch.arange(0, w)[None]
        y1d = torch.arange(0, h)[:, None]
        x2d = x1d.repeat([h, 1]).to(depth1)  # (h, w)
        y2d = y1d.repeat([1, w]).to(depth1)  # (h, w)
        ones_2d = torch.ones(size=(h, w)).to(depth1)  # (h, w)
        ones_4d = ones_2d[None, :, :, None, None].repeat([b, 1, 1, 1, 1])  # (b, h, w, 1, 1)
        pos_vectors_homo = torch.stack([x2d, y2d, ones_2d], dim=2)[None, :, :, :, None]  # (1, h, w, 3, 1)

        intrinsic1_inv = torch.linalg.inv(intrinsic1)  # (b, 3, 3)
        intrinsic1_inv_4d = intrinsic1_inv[:, None, None]  # (b, 1, 1, 3, 3)
        intrinsic2_4d = intrinsic2[:, None, None]  # (b, 1, 1, 3, 3)
        depth_4d = depth1[:, 0][:, :, :, None, None]  # (b, h, w, 1, 1)
        trans_4d = transformation[:, None, None]  # (b, 1, 1, 4, 4)

        unnormalized_pos = torch.matmul(intrinsic1_inv_4d, pos_vectors_homo)  # (b, h, w, 3, 1)
        world_points = depth_4d * unnormalized_pos  # (b, h, w, 3, 1)
        world_points_homo = torch.cat([world_points, ones_4d], dim=3)  # (b, h, w, 4, 1)
        trans_world_homo = torch.matmul(trans_4d, world_points_homo)  # (b, h, w, 4, 1)
        trans_world = trans_world_homo[:, :, :, :3]  # (b, h, w, 3, 1)
        trans_norm_points = torch.matmul(intrinsic2_4d, trans_world)  # (b, h, w, 3, 1)
        return trans_norm_points

    def bilinear_splatting(self, frame1: torch.Tensor, mask1: Optional[torch.Tensor], depth1: torch.Tensor,
                           flow12: torch.Tensor, flow12_mask: Optional[torch.Tensor], is_image: bool = False) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting
        :param frame1: (b,c,h,w)
        :param mask1: (b,1,h,w): 1 for known, 0 for unknown. Optional
        :param depth1: (b,1,h,w)
        :param flow12: (b,2,h,w)
        :param flow12_mask: (b,1,h,w): 1 for valid flow, 0 for invalid flow. Optional
        :param is_image: if true, output will be clipped to (-1,1) range
        :return: warped_frame2: (b,c,h,w)
                 mask2: (b,1,h,w): 1 for known and 0 for unknown
        """
        if self.resolution is not None:
            assert frame1.shape[2:4] == self.resolution
        b, c, h, w = frame1.shape
        if mask1 is None:
            mask1 = torch.ones(size=(b, 1, h, w)).to(frame1)
        if flow12_mask is None:
            flow12_mask = torch.ones(size=(b, 1, h, w)).to(flow12)
        grid = self.create_grid(b, h, w).to(frame1)
        trans_pos = flow12 + grid

        trans_pos_offset = trans_pos + 1
        trans_pos_floor = torch.floor(trans_pos_offset).long()
        trans_pos_ceil = torch.ceil(trans_pos_offset).long()
        trans_pos_offset = torch.stack([
            torch.clamp(trans_pos_offset[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_offset[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_floor = torch.stack([
            torch.clamp(trans_pos_floor[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_floor[:, 1], min=0, max=h + 1)], dim=1)
        trans_pos_ceil = torch.stack([
            torch.clamp(trans_pos_ceil[:, 0], min=0, max=w + 1),
            torch.clamp(trans_pos_ceil[:, 1], min=0, max=h + 1)], dim=1)

        prox_weight_nw = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_sw = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_offset[:, 0:1] - trans_pos_floor[:, 0:1]))
        prox_weight_ne = (1 - (trans_pos_offset[:, 1:2] - trans_pos_floor[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))
        prox_weight_se = (1 - (trans_pos_ceil[:, 1:2] - trans_pos_offset[:, 1:2])) * \
                         (1 - (trans_pos_ceil[:, 0:1] - trans_pos_offset[:, 0:1]))

        sat_depth1 = torch.clamp(depth1, min=0, max=1000)
        log_depth1 = torch.log(1 + sat_depth1)
        depth_weights = torch.exp(log_depth1 / log_depth1.max() * 50)

        weight_nw = torch.moveaxis(prox_weight_nw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_sw = torch.moveaxis(prox_weight_sw * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_ne = torch.moveaxis(prox_weight_ne * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])
        weight_se = torch.moveaxis(prox_weight_se * mask1 * flow12_mask / depth_weights, [0, 1, 2, 3], [0, 3, 1, 2])

        warped_frame = torch.zeros(size=(b, h + 2, w + 2, c), dtype=torch.float32).to(frame1)
        warped_weights = torch.zeros(size=(b, h + 2, w + 2, 1), dtype=torch.float32).to(frame1)

        frame1_cl = torch.moveaxis(frame1, [0, 1, 2, 3], [0, 3, 1, 2])
        batch_indices = torch.arange(b)[:, None, None].to(frame1.device)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_nw), accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_sw), accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_ne), accumulate=True)
        warped_frame.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                torch.einsum('bhwc,bhwk->bhwc', frame1_cl, weight_se), accumulate=True)

        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_floor[:, 0]),
                                  weight_nw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_floor[:, 0]),
                                  weight_sw, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_floor[:, 1], trans_pos_ceil[:, 0]),
                                  weight_ne, accumulate=True)
        warped_weights.index_put_((batch_indices, trans_pos_ceil[:, 1], trans_pos_ceil[:, 0]),
                                  weight_se, accumulate=True)

        warped_frame_cf = torch.moveaxis(warped_frame, [0, 1, 2, 3], [0, 2, 3, 1])
        warped_weights_cf = torch.moveaxis(warped_weights, [0, 1, 2, 3], [0, 2, 3, 1])
        cropped_warped_frame = warped_frame_cf[:, :, 1:-1, 1:-1]
        cropped_weights = warped_weights_cf[:, :, 1:-1, 1:-1]

        mask = cropped_weights > 0
        zero_value = -1 if is_image else 0
        zero_tensor = torch.tensor(zero_value, dtype=frame1.dtype, device=frame1.device)
        warped_frame2 = torch.where(mask, cropped_warped_frame / cropped_weights, zero_tensor)
        mask2 = mask.to(frame1)

        if is_image:
            assert warped_frame2.min() >= -1.1  # Allow for rounding errors
            assert warped_frame2.max() <= 1.1
            warped_frame2 = torch.clamp(warped_frame2, min=0, max=1)
        return warped_frame2, mask2

    @staticmethod
    def create_grid(b, h, w):
        x_1d = torch.arange(0, w)[None]
        y_1d = torch.arange(0, h)[:, None]
        x_2d = x_1d.repeat([h, 1])
        y_2d = y_1d.repeat([1, w])
        grid = torch.stack([x_2d, y_2d], dim=0)
        batch_grid = grid[None].repeat([b, 1, 1, 1])
        return batch_grid

    @staticmethod
    def get_device(device: str):
        """
        Returns torch device object
        :param device: cpu/gpu0/gpu1
        :return:
        """
        if device == 'cpu':
            device = torch.device('cpu')
        elif device.startswith('gpu') and torch.cuda.is_available():
            gpu_num = int(device[3:])
            device = torch.device(f'cuda:{gpu_num}')
        else:
            device = torch.device('cpu')
        return device



    def bidirectional_warp(self, img1: torch.Tensor, depth_map1: torch.Tensor,
                      w2c1: torch.Tensor, w2c2: torch.Tensor,
                      K1: torch.Tensor, K2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward warping 完整流程实现（返回虚拟视图和有效掩码）
        Args:
            img1: (b, 3, h, w) 参考视图图像
            depth_map1: (b, 1, h, w) 参考视图深度图
            w2c1: (b, 4, 4) 参考视图的世界到相机变换矩阵
            w2c2: (b, 4, 4) 虚拟视图的世界到相机变换矩阵
            K1: (b, 3, 3) 参考视图内参
            K2: (b, 3, 3) 虚拟视图内参
        Returns:
            img2: (b, 3, h, w) 生成的虚拟视图图像
            valid_mask: (b, 1, h, w) 有效区域掩码（1=有效，0=无效）
        """
        b, _, h, w = img1.shape

        # 1. 获取参考视图的像素坐标
        pixel1 = self.get_pixel_coords(b, h, w).to(img1.device)

        # 2. 反投影到3D空间
        point1_cam1 = self.unproject(pixel1, depth_map1, K1)

        # 3. 转换到虚拟视图坐标系
        point1_cam2 = self.transform_points(point1_cam1, w2c1, w2c2)

        # 4. 投影到虚拟视图成像平面
        pixel12, depths = self.project(point1_cam2, K2)

        # 5. 生成初始有效掩码（检查投影点是否在图像范围内）
        valid_mask = (pixel12[..., 0] >= 0) & (pixel12[..., 0] <= 1) & (pixel12[..., 1] >= 0) & (pixel12[..., 1] <= 1)
        valid_mask = valid_mask.float().unsqueeze(1)  # (b, 1, h, w)

        # 6. 初始化虚拟视图的深度图
        depth_map2 = torch.zeros((b, 1, h, w), device = img1.device)

        # 7. 分散深度值并更新有效掩码
        depth_map2, updated_mask = self.scatter_depth_values_with_mask(
            pixel12, depths, valid_mask, depth_map2
        )

        # 8. 对depth_map2进行空洞填充
        depth_map2_filled = self.fill_holes(depth_map2)
        # depth_map2_filled = insert_depth32f(depth_map2_filled)
        # depth_map2_filled = depth_map2

        # 9. 将虚拟视图像素反投影到3D空间
        pixel2 = self.get_pixel_coords(b, h, w).to(img1.device)
        point2_cam2 = self.unproject(pixel2, depth_map2_filled, K2)

        # 10. 转换到参考视图坐标系
        point2_cam1 = self.transform_points(point2_cam2, w2c2, w2c1)

        # 11. 投影到参考视图成像平面
        pixel1_reproj, _ = self.project(point2_cam1, K1)

        # 12. 双线性插值获取虚拟视图图像
        img2 = self.bilinear_sample(img1, pixel1_reproj)

        # 13. 最终有效掩码（结合初始有效性和填充后的深度）
        # 使用torch.logical_and处理浮点掩码
        final_valid_mask = torch.logical_and(
            updated_mask > 0.5,  # 将float掩码转为bool
            depth_map2_filled > 0
        ).float()

        return img2, depth_map2_filled, final_valid_mask

    def scatter_depth_values_with_mask(self, src_coords: torch.Tensor, src_depths: torch.Tensor,
                                       valid_mask: torch.Tensor, target_depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h_src, w_src, _ = src_coords.shape
        _, _, H_tgt, W_tgt = target_depth.shape

        # 确保深度图形状正确 (b, h_src, w_src, 1)
        src_depths = src_depths.permute(0, 2, 3, 1) if src_depths.dim() == 4 else src_depths.unsqueeze(-1)

        # 转换为像素坐标
        pixel_coords = src_coords * torch.tensor([W_tgt - 1, H_tgt - 1], device=src_coords.device)

        # 计算四个邻近像素
        x = pixel_coords[..., 0]
        y = pixel_coords[..., 1]
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1

        # 计算权重 (b, h_src, w_src, 4)
        wx1, wy1 = x - x0.float(), y - y0.float()
        wx0, wy0 = 1 - wx1, 1 - wy1
        weights = torch.stack([wx0 * wy0, wx0 * wy1, wx1 * wy0, wx1 * wy1], dim=-1)

        # 四个角的坐标 (b, h_src, w_src, 4, 2)
        coords = torch.stack([
            torch.stack([x0, y0], dim=-1),  # 左上
            torch.stack([x0, y1], dim=-1),  # 左下
            torch.stack([x1, y0], dim=-1),  # 右上
            torch.stack([x1, y1], dim=-1)  # 右下
        ], dim=-2)

        # 初始化输出
        new_depth = torch.zeros_like(target_depth)
        new_mask = torch.zeros_like(target_depth)
        weight_sum = torch.zeros_like(target_depth)

        # 批量索引 (b, h_src, w_src, 1)
        batch_idx = torch.arange(b, device=src_coords.device)[:, None, None, None].expand(-1, h_src, w_src, -1)

        # 对每个角点进行处理
        for i in range(4):
            curr_coords = coords[..., i, :]  # (b, h_src, w_src, 2)
            curr_weights = weights[..., i] * valid_mask.squeeze(1)  # (b, h_src, w_src)

            # 边界检查
            valid = (curr_coords[..., 0] >= 0) & (curr_coords[..., 0] < W_tgt) & \
                    (curr_coords[..., 1] >= 0) & (curr_coords[..., 1] < H_tgt)

            # 展平所有张量
            flat_b = batch_idx[valid].squeeze(-1)  # (n_valid,)
            flat_y = curr_coords[..., 1][valid]  # (n_valid,)
            flat_x = curr_coords[..., 0][valid]  # (n_valid,)
            flat_weights = curr_weights[valid]  # (n_valid,)
            flat_depths = src_depths[valid]  # (n_valid, 1)

            # 只处理有有效索引的情况
            if flat_b.numel() > 0:
                # 使用index_put_进行原子操作
                new_depth.index_put_(
                    (flat_b, torch.zeros_like(flat_b), flat_y, flat_x),
                    flat_depths.squeeze(-1) * flat_weights,
                    accumulate=True
                )
                weight_sum.index_put_(
                    (flat_b, torch.zeros_like(flat_b), flat_y, flat_x),
                    flat_weights,
                    accumulate=True
                )
                new_mask.index_put_(
                    (flat_b, torch.zeros_like(flat_b), flat_y, flat_x),
                    torch.ones_like(flat_weights),
                    accumulate=True
                )

        # 归一化深度值
        new_depth = new_depth / (weight_sum + 1e-6)
        new_mask = (new_mask > 0).float()

        return new_depth, new_mask


    def get_pixel_coords(self, b: int, h: int, w: int) -> torch.Tensor:
        """生成像素坐标网格 (b, h, w, 2)"""
        x = torch.arange(w, dtype=torch.float32).view(1, 1, -1)  # (1, 1, w)
        y = torch.arange(h, dtype=torch.float32).view(1, -1, 1)  # (1, h, 1)

        # 修正后的repeat操作 - 保持3维
        x_expanded = x.repeat(1, h, 1)  # (1, h, w)
        y_expanded = y.repeat(1, 1, w)  # (1, h, w)

        grid = torch.stack([x_expanded, y_expanded], dim=-1)  # (1, h, w, 2)
        return grid.repeat(b, 1, 1, 1)  # (b, h, w, 2)

    def unproject(self, pixel_coords: torch.Tensor, depth: torch.Tensor,
                  K: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = pixel_coords.shape

        # 转换为齐次坐标
        ones = torch.ones_like(pixel_coords[..., :1])
        pixel_coords_homo = torch.cat([pixel_coords, ones], dim=-1)  # (b, h, w, 3)

        # 反投影到相机坐标系
        K_inv = torch.linalg.inv(K)
        points_cam = torch.einsum('bij,bhwj->bhwi', K_inv, pixel_coords_homo)  # (b, h, w, 3)

        # 乘以深度
        points_cam = points_cam * depth.permute(0, 2, 3, 1)  # (b, h, w, 3)

        return points_cam

    def transform_points(self, points: torch.Tensor,
                         w2c_src: torch.Tensor, w2c_tgt: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = points.shape

        # 转换为齐次坐标
        ones = torch.ones_like(points[..., :1])
        points_homo = torch.cat([points, ones], dim=-1)  # (b, h, w, 4)

        # 计算变换矩阵: c2_tgt * c2_src^-1
        c2w_src = torch.linalg.inv(w2c_src)
        transform = torch.bmm(w2c_tgt, c2w_src)  # (b, 4, 4)

        # 应用变换
        transform = transform.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, 4, 4)
        transformed_points = torch.einsum('bijkl,bhwl->bhwk', transform, points_homo)  # (b, h, w, 4)

        return transformed_points[..., :3]  # 去掉齐次坐标

    def project(self, points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = points.shape

        # 投影到成像平面
        projected = torch.einsum('bij,bhwj->bhwi', K, points)  # (b, h, w, 3)
        pixel_coords = projected[..., :2] / (projected[..., 2:3] + 1e-6)  # (b, h, w, 2)



        # 归一化到[0,1]
        pixel_coords[..., 0] = pixel_coords[..., 0] / (w - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (h - 1)

        # 获取深度
        depth = projected[..., 2:3].permute(0, 3, 1, 2)  # (b, 1, h, w)

        return pixel_coords, depth

    def fill_holes(self, depth: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        valid_mask = (depth > 0).float()
        filled_depth = depth.clone()

        # 使用最大池化填充空洞
        for _ in range(3):  # 多次迭代确保填充
            max_pool = F.max_pool2d(filled_depth * valid_mask,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=kernel_size // 2)
            new_valid = F.max_pool2d(valid_mask,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size // 2)

            update_mask = (valid_mask < 1) & (new_valid > 0)
            filled_depth = torch.where(update_mask, max_pool / (new_valid + 1e-6), filled_depth)
            valid_mask = torch.clamp(valid_mask + update_mask.float(), 0, 1)

        return filled_depth

    def bilinear_sample(self, img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # 将坐标从[0,1]映射到[-1,1]
        coords = coords * 2 - 1

        # 使用grid_sample进行双线性插值
        sampled = F.grid_sample(img, coords, mode='bilinear', padding_mode='zeros', align_corners=False)
        return sampled
