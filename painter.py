import os
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from easydict import EasyDict as edict
from torch.optim.lr_scheduler import LambdaLR
# from torchsummary import summary
import pydiffvg
import utils
import xml.etree.ElementTree as ET
from models import PointMLP

class Painter(torch.nn.Module):
    def __init__(self,
                 args,
                 svg_path: str,
                 num_frames: int,
                 device,
                 path_to_trained_mlp=None):
        super(Painter, self).__init__()
        self.caption = args.caption
        self.svg_path = os.path.join(args.target, "postprocess")
        self.motion_path = args.target
        self.output_folder = args.output_folder
        self.num_frames = num_frames
        self.device = device
        self.optim_points = args.optim_points
        self.optim_points_with_mlp = args.optim_points_with_mlp
        self.render = pydiffvg.RenderFunction.apply
        self.normalize_input = args.normalize_input
        self.use_canvas = args.use_canvas
        self.dino_fusion = args.dino
        self.timeE = args.timeE
        self.merge_groups = args.merge_groups  
        # initialize SVG shapes
        self.init_shapes()
        self.mlp_points_by_g = {}
        self.points_mlp_input_by_g_ = {}

        if self.optim_points_with_mlp:
            for g_idx, points_mlp_input in self.points_mlp_input_by_g.items():
                self.points_mlp_input_by_g_[g_idx] = points_mlp_input.unsqueeze(0).to(device)
                # Initialize a PointMLP network to predict per-point offsets
                # input_dim: input dimension, equals the flattened total count of all point coordinates (e.g., 2496)
                # inter_dim: hidden dimension of the MLP (e.g., 128)
                # num_points_per_frame: number of points per frame (e.g., 52)
                # num_frames: total number of frames (e.g., 24)
                # predict_global_frame_deltas: whether to predict a global delta per frame; value acts as the output weight
                # predict_only_global: whether to only predict global deltas
                # rotation_weight: weight for the 'rotation' terms in the global transform matrix
                # scale_weight: weight for the 'scale' terms in the global transform matrix  self.counts_points_per_frame_by_g
                # shear_weight: weight for the 'shear' terms in the global transform matrix
                # translation_weight: weight for the 'translation' terms in the global transform matrix
                self.mlp_points_by_g[g_idx] = PointMLP(input_dim=torch.numel(self.points_mlp_input_by_g[g_idx]),
                                            inter_dim=args.inter_dim,
                                            num_points_per_frame=self.counts_points_per_frame_by_g[g_idx],
                                            num_frames=num_frames,
                                            device=device,
                                            predict_global_frame_deltas=args.predict_global_frame_deltas,
                                            predict_only_global=args.predict_only_global, 
                                            rotation_weight=args.rotation_weight, 
                                            scale_weight=args.scale_weight, 
                                            shear_weight=args.shear_weight,
                                            translation_weight=args.translation_weight,                                    
                                            output_folder=self.output_folder,
                                            dino_fusion=self.dino_fusion, timeE=self.timeE).to(device)

            for g_idx, points_mlp_input in self.points_mlp_input_by_g.items():
                if path_to_trained_mlp:
                    print(f"Loading MLP from {path_to_trained_mlp}")
                    self.mlp_points_by_g[g_idx].load_state_dict(torch.load(path_to_trained_mlp))
                    self.mlp_points_by_g[g_idx].eval()

            # Init the weights of LayerNorm for global translation MLP if needed.
            if args.translation_layer_norm_weight:
                self.init_translation_norm(args.translation_layer_norm_weight)       
                   
    def get_motion_data_from_canvas(self, g_idx):
        
        motion_data_path_src = os.path.join(self.motion_path, f'motion_data_{g_idx}.txt')
        motion_data_path = os.path.join(self.output_folder, f'motion_data_{g_idx}.txt')
        with open(motion_data_path_src, 'rb') as src_file:
            with open(motion_data_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
                
        if os.path.exists(motion_data_path):
            print(f"√find motion data from canvas: {motion_data_path}")
            # read motion data from cache
            motion_data = []
            with open(motion_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # extract numbers from "frame 1: (0, -5)"
                    import re
                    numbers = re.findall(r'-?\d+', line)
                    if len(numbers) >= 2:  
                        motion_data.append((int(numbers[-2]), int(numbers[-1])))
            if len(motion_data) == self.num_frames:
                return torch.tensor(motion_data, dtype=torch.float32)
            print(f"data frame number mismatch, need {self.num_frames} frames, actual {len(motion_data)} frames")
         
    def init_shapes(self):
        """load shapes from svg file and initialize data structure"""
        # 1. read SVG file
        svg_path = f'{self.svg_path}.svg'
        self.canvas_width, self.canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg_path)

        # 2. initialize basic data
        # count paths and points per g
        self.count_paths_per_g_per_frame = self.count_paths_per_g(svg_path)
        self.num_g = len(self.count_paths_per_g_per_frame)
        self.counts_points_per_frame_by_g = {
            g_idx: num_paths * 4  # each path has 4 control points
            for g_idx, num_paths in self.count_paths_per_g_per_frame.items()
        }
        # 3. calculate center point
        center, all_points = get_center_of_mass(shapes_init)
        self.original_center = center.clone().to(self.device)
        self.original_center.requires_grad = False
        #4. get motion data from canvas
        motion_data_by_g = {}
        if self.use_canvas:
            for g_idx in range(self.num_g):
                motion_data = self.get_motion_data_from_canvas(g_idx)
                if motion_data is not None:
                    motion_data_by_g[g_idx] = motion_data.to(self.device)
                    
        # 4. calculate deltas from center
        deltas_from_center = get_deltas(all_points, center, self.device)
        # 5. initialize data structure
        self.frames_shapes_by_g = {}
        self.frames_shapes_group_by_g = {}
        self.frames_xy_deltas_from_center_by_g = {}
        self.points_mlp_input_by_g = {}
        
        # 6. process data by group (g)
        cur = 0
        for g_idx, num_paths in self.count_paths_per_g_per_frame.items():
            frames_shapes = []
            frames_shapes_group = []
            frames_xy_deltas_from_center = []

            for frame_idx in range(self.num_frames):
                # get base deltas
                base_deltas = deltas_from_center[cur:cur+num_paths]
                frame_shapes = []
                # apply motion data from canvas
                if g_idx in motion_data_by_g:
                    frame_motion = motion_data_by_g[g_idx][frame_idx]  # get motion data for current frame
                    # create new list to store updated deltas
                    frame_deltas = []
                    
                    # process each path
                    for path_idx, path_delta in enumerate(base_deltas):
                        # apply motion to delta
                        motion = frame_motion.unsqueeze(0).expand(4, -1)  # [4, 2]

                        updated_delta = path_delta + motion
                        frame_deltas.append(updated_delta)
                        
                        # update actual path points coordinates
                        original_path = shapes_init[cur + path_idx]
                        new_points = updated_delta + self.original_center  # delta + center = actual coordinates
                        
                        # create new Path object
                        new_path = pydiffvg.Path(
                            num_control_points=original_path.num_control_points,
                            points=new_points,
                            stroke_width=original_path.stroke_width,
                            is_closed=original_path.is_closed
                        )
                        frame_shapes.append(new_path)
                    
                    frames_xy_deltas_from_center.append(frame_deltas)
                else:
                    # if not using LLM
                    frames_xy_deltas_from_center.append(base_deltas)
                    frame_shapes = shapes_init[cur:cur+num_paths]
                
                frames_shapes.append(frame_shapes)
                frames_shapes_group.append(shape_groups_init[cur:cur+num_paths])
            
            self.frames_shapes_by_g[g_idx] = frames_shapes
            self.frames_shapes_group_by_g[g_idx] = frames_shapes_group
            self.frames_xy_deltas_from_center_by_g[g_idx] = frames_xy_deltas_from_center
            
            # prepare Network input
            tensor_points_init = [torch.cat(frames_xy_deltas_from_center[i])
                              for i in range(self.num_frames)]
            self.points_mlp_input_by_g[g_idx] = torch.cat(tensor_points_init)
            
            cur += num_paths
        
    def render_frames_to_tensor_mlp(self):
        """render frames sequence to tensor, support merging all <g> groups"""
        frames_shepesAndgroups_by_g = {}
        points_init_frame_by_g = {}
        
        # apply motion_data to each <g> group
        for g_idx in self.points_mlp_input_by_g.keys():
            frames_shepesAndgroups = []
            all_new_points = []
            
            # get initial point coordinates and apply MLP prediction
            prev_points = self.points_mlp_input_by_g[g_idx].clone().squeeze(0) + self.original_center
            frame_input = self.points_mlp_input_by_g_[g_idx]
            if self.normalize_input:
                frame_input = utils.normalize_tensor(frame_input)
            delta_prediction = self.mlp_points_by_g[g_idx](frame_input)
            
            # process each frame
            for i in range(self.num_frames):
                shapes, shapes_groups = self.frames_shapes_by_g[g_idx][i], self.frames_shapes_group_by_g[g_idx][i]
                new_shapes, new_shape_groups, frame_new_points = [], [], []
                
                # calculate current frame point coordinates
                start_frame_slice = i * self.counts_points_per_frame_by_g[g_idx]
                point_delta_leanred_cur_frame = delta_prediction[
                    start_frame_slice: start_frame_slice + self.counts_points_per_frame_by_g[g_idx], :]
                points_cur_frame = prev_points[
                    start_frame_slice: start_frame_slice + self.counts_points_per_frame_by_g[g_idx], :] + point_delta_leanred_cur_frame
                
                # create new Path and ShapeGroup objects for each shape
                counter = 0
                for j in range(len(shapes)):
                    shape, shapes_group = shapes[j], shapes_groups[j]
                    points_vars = shape.points.clone()
                    
                    # update point coordinates
                    points_vars[:, 0] = points_cur_frame[counter:counter + shape.points.shape[0], 0]
                    points_vars[:, 1] = points_cur_frame[counter:counter + shape.points.shape[0], 1]
                    counter += shape.points.shape[0]
                    
                    # save and create new objects
                    frame_new_points.append(points_vars.to(self.device))
                    path = pydiffvg.Path(
                        num_control_points=shape.num_control_points, 
                        points=points_vars,
                        stroke_width=shape.stroke_width, 
                        is_closed=shape.is_closed
                    )
                    new_shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([len(new_shapes) - 1]),
                        fill_color=shapes_group.fill_color,
                        stroke_color=torch.tensor([0, 0, 0, 1])
                    )
                    new_shape_groups.append(path_group)
                
                # save current frame result
                frames_shepesAndgroups.append((new_shapes, new_shape_groups))
                all_new_points.append(frame_new_points)
            
            frames_shepesAndgroups_by_g[g_idx] = frames_shepesAndgroups
            points_init_frame_by_g[g_idx] = all_new_points
        
        # return form according to merge_groups switch
        if self.merge_groups:
            # merge processing
            composed_frames_tensor = self.render_combined_frames(frames_shepesAndgroups_by_g)
            merged_frames_shepesAndgroups = self.merge_frames_shapes_and_groups(frames_shepesAndgroups_by_g)
            merged_points_init_frame = self.merge_points_init_frame(points_init_frame_by_g)

            return composed_frames_tensor, merged_frames_shepesAndgroups, merged_points_init_frame
        else:
            composed_frames_tensor = self.render_combined_frames(frames_shepesAndgroups_by_g)
            return composed_frames_tensor, frames_shepesAndgroups_by_g, points_init_frame_by_g

    def merge_frames_shapes_and_groups(self, frames_shepesAndgroups_by_g):
        """merge all groups' shapes and shape_groups into a single list"""
        merged_frames_shepesAndgroups = []
        
        # merge each frame
        for frame_idx in range(self.num_frames):
            all_shapes = []
            all_shape_groups = []
            shape_counter = 0
            
            # traverse each g group
            for g_idx in sorted(frames_shepesAndgroups_by_g.keys()):
                shapes, shape_groups = frames_shepesAndgroups_by_g[g_idx][frame_idx]
                
                # add shapes
                all_shapes.extend(shapes)
                
                # update shape_groups' shape_ids and add
                for group in shape_groups:
                    new_group = pydiffvg.ShapeGroup(
                        shape_ids=group.shape_ids + shape_counter,
                        fill_color=group.fill_color,
                        stroke_color=group.stroke_color
                    )
                    all_shape_groups.append(new_group)
                
                shape_counter += len(shapes)
            
            # add merged frames
            merged_frames_shepesAndgroups.append((all_shapes, all_shape_groups))
        
        return merged_frames_shepesAndgroups

    def merge_points_init_frame(self, points_init_frame_by_g):
        """merge all groups' point coordinates"""
        merged_points_init_frame = []
        
        # get number of frames for each group
        max_frames = max([len(points) for points in points_init_frame_by_g.values()])
        
        # merge each frame's point coordinates
        for frame_idx in range(max_frames):
            frame_points = []
            for g_idx in sorted(points_init_frame_by_g.keys()):
                if frame_idx < len(points_init_frame_by_g[g_idx]):
                    frame_points.extend(points_init_frame_by_g[g_idx][frame_idx])
            merged_points_init_frame.append(frame_points)
        
        return merged_points_init_frame
   
    def render_combined_frames(self, frames_shepesAndgroups_by_g):
        """merge all g groups' shapes and shape_groups and render and save"""
        frames_combined = []
        
        # traverse each frame
        for frame_idx in range(self.num_frames):
            all_shapes = []
            all_shape_groups = []
            shape_counter = 0
            
            # traverse each g group
            for g_idx in frames_shepesAndgroups_by_g.keys():
                # get shapes and shape_groups of the frame_idx frame
                shapes, shape_groups = frames_shepesAndgroups_by_g[g_idx][frame_idx]  
                
                # add shapes
                all_shapes.extend(shapes)
                
                # update shape_groups' shape_ids and add
                for i, group in enumerate(shape_groups):
                    new_group = pydiffvg.ShapeGroup(
                        shape_ids=group.shape_ids + shape_counter,
                        fill_color=group.fill_color,
                        stroke_color=group.stroke_color
                    )
                    all_shape_groups.append(new_group)
                
                shape_counter += len(shapes)
            
            # render merged frames
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.canvas_width, 
                self.canvas_height,
                all_shapes,
                all_shape_groups
            )
            
            cur_im = self.render(
                self.canvas_width, 
                self.canvas_height,
                2, 2, 0, None, 
                *scene_args
            )
            
            # process alpha channel
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                    torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            
            frames_combined.append(cur_im)
        
        return torch.stack(frames_combined)
    
    def render_frames_to_tensor_direct_optim(self):
        # the effect is bad so pass the function 
        frames_init, frames_shepesAndgroups, points_init_frame = [], [], []

        # traverse each frame
        for i in range(self.num_frames):
            # get shapes and shape_groups of the current frame
            shapes = self.frames_shapes[i]
            shapes_groups = self.frames_shapes_group[i]
            new_shapes, new_shape_groups = [], []

            # get the offset of the current frame relative to the center point
            deltas_from_center_cur_frame = self.frames_xy_deltas_from_center[i]
            
            # traverse each shape in the current frame
            for j in range(len(shapes)):
                shape, shapes_group = shapes[j], shapes_groups[j]
                point_delta_leanred = deltas_from_center_cur_frame[j]
                points_vars = shape.points.clone()

                # update point coordinates: new coordinates = learned offset + original center point coordinates
                points_vars[:, 0] = point_delta_leanred[:, 0] + self.original_center[0]  # x coordinates
                points_vars[:, 1] = point_delta_leanred[:, 1] + self.original_center[1]  # y coordinates
                
                # save initial point coordinates only for the first frame
                if i == 0:
                    points_init_frame.append(points_vars)

                # create new Path objects
                path = pydiffvg.Path(
                    num_control_points=shape.num_control_points, points=points_vars,
                    stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                new_shapes.append(path)
                
                # create new ShapeGroup objects
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(new_shapes) - 1]),
                    fill_color=shapes_group.fill_color,
                    stroke_color=torch.tensor([0, 0, 0, 1]))  
                new_shape_groups.append(path_group)

            # serialize scene parameters
            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes,
                                                                 new_shape_groups)
            # render current frame
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)

            # process alpha channel: convert RGBA to RGB
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            
            # save rendered result and SVG data
            frames_init.append(cur_im)
            frames_shepesAndgroups.append((new_shapes, new_shape_groups))
            
        return torch.stack(frames_init), frames_shepesAndgroups, points_init_frame

    def render_frames_to_tensor(self, mlp=True):
        """render frames sequence to tensor, support merge_groups switch"""
        if self.optim_points_with_mlp and mlp:
            return self.render_frames_to_tensor_mlp()
        else:
            frames_init, frames_shepesAndgroups, points_init_frame = self.render_frames_to_tensor_direct_optim()
            return frames_init, frames_shepesAndgroups, points_init_frame
  
    def get_points_params(self):
        if self.optim_points_with_mlp:
            composed_points_params = []
            for g_idx, mlp_points in self.mlp_points_by_g.items():
                composed_points_params.append(mlp_points.get_points_params())
            return composed_points_params
        return self.parameters_["point_delta"]


    def get_global_params(self):
        composed_global_params = []
        for g_idx, mlp_points in self.mlp_points_by_g.items():
            composed_global_params.append(mlp_points.get_global_params())
        return composed_global_params

    def log_state(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        torch.save(self.mlp_points.state_dict(), f"{output_path}/model.pt")
        print(f"Model saved to {output_path}/model.pt")

    def init_translation_norm(self, translation_layer_norm_weight):
        print(f"Initializing translation layerNorm to {translation_layer_norm_weight}")
        for g_idx, mlp_points in self.mlp_points_by_g.items():
            for child in mlp_points.frames_rigid_translation.children():
                if isinstance(child, nn.LayerNorm):
                    with torch.no_grad():
                        child.weight *= translation_layer_norm_weight

    def count_paths_per_g(self, svg_path):
        """count the number of paths in each <g> group in the SVG file
        
        Args:
            svg_path (str): path of the SVG file
            
        Returns:
            dict: key is the group index (starting from 0), value is the number of paths in the group
        """
        # parse the SVG file
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # initialize the counting dictionary
        paths_count = {}
        
        # traverse all g tags
        for g_idx, g_elem in enumerate(root.findall('.//{http://www.w3.org/2000/svg}g')):
            # calculate the number of paths in the current g tag
            path_count = len(g_elem.findall('{http://www.w3.org/2000/svg}path'))
            paths_count[g_idx] = path_count

        return paths_count



    def save_frames_debug_images(self, g_idx, frames_shapes):
        """save the rendered images of all frames of the specified group, for debugging"""
        # create the save directory
        debug_dir = os.path.join(self.output_folder, f'debug_frames_g{g_idx}')
        os.makedirs(debug_dir, exist_ok=True)
        
        # add debug print
        print(f"\n=== Debug Info for g_idx={g_idx} ===")
        print(f"Total frames: {len(frames_shapes)}")
        
        # render each frame
        for frame_idx, frame_shapes in enumerate(frames_shapes):
            if len(frame_shapes) == 0:
                print(f"Warning: No shapes in frame {frame_idx}")
                continue
            
            # get the corresponding shape_groups and ensure shape_ids are correct
            frame_groups = []
            for i, group in enumerate(self.frames_shapes_group_by_g[g_idx][frame_idx]):
                # create new ShapeGroup, ensure shape_ids are correct
                new_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([i]),  # use current index
                    fill_color=group.fill_color,
                    stroke_color=group.stroke_color
                )
                frame_groups.append(new_group)
            
            try:
                scene_args = pydiffvg.RenderFunction.serialize_scene(
                    self.canvas_width, 
                    self.canvas_height,
                    frame_shapes,
                    frame_groups
                )
                
                # render
                img = self.render(
                    self.canvas_width, 
                    self.canvas_height,
                    2, 2, 0, None, 
                    *scene_args
                )
                
                # process alpha channel
                img = img[:, :, 3:4] * img[:, :, :3] + \
                      torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - img[:, :, 3:4])
                img = img[:, :, :3]
                
                # convert to numpy array and save
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                save_path = os.path.join(debug_dir, f'frame_{frame_idx:03d}.png')
                import cv2
                cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                
            except Exception as e:
                print(f"Error in frame {frame_idx}:")
                print(f"frame_shapes: {frame_shapes}")
                print(f"frame_groups: {frame_groups}")
                raise e
        
        print(f"saved all frames of group {g_idx} to {debug_dir}")



class PainterOptimizer:
    def __init__(self, args):
        # self.painter = painter
        self.lr_local = args.lr_local
        self.lr_base_global = args.lr_base_global
        self.lr_init = args.lr_init
        self.lr_final = args.lr_final
        self.lr_delay_mult = args.lr_delay_mult
        self.lr_delay_steps = args.lr_delay_steps
        self.max_steps = args.num_iter
        self.lr_lambda = lambda step: self.learning_rate_decay(step) / self.lr_init
        self.optim_points = args.optim_points
        self.optim_global = args.split_global_loss

    def learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def init_optimizers(self, composed_global_params, composed_points_params):
        if self.optim_global:
            combined_global_params = itertools.chain(*composed_global_params)
            self.global_delta_optimizer = torch.optim.Adam(combined_global_params, lr=self.lr_base_global,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_global = LambdaLR(self.global_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

        if self.optim_points:
            combined_points_params = itertools.chain(*composed_points_params)
            self.points_delta_optimizer = torch.optim.Adam(combined_points_params, lr=self.lr_local,
                                                           betas=(0.9, 0.9), eps=1e-6)
            self.scheduler_points = LambdaLR(self.points_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_lr(self):
        if self.optim_global:
            self.scheduler_global.step()
        if self.optim_points:
            self.scheduler_points.step()

    def zero_grad_(self):
        if self.optim_points:
            self.points_delta_optimizer.zero_grad()
        # if self.optim_global:
        #     self.global_delta_optimizer.zero_grad()


    def step_(self, skip_global=False, skip_points=False):
        if self.optim_global and not skip_global:
            self.global_delta_optimizer.step()
        if self.optim_points and not skip_points:
            self.points_delta_optimizer.step()

    def get_lr(self, optim="points"):
        if optim == "points" and self.optim_points:
            return self.points_delta_optimizer.param_groups[0]['lr']
        else:
            return None
        
    def get_lr(self, optim="points"):
        if optim == "points" and self.optim_points:
            return self.points_delta_optimizer.param_groups[0]['lr']
        elif optim == "global" and self.optim_global:
            return self.global_delta_optimizer.param_groups[0]['lr']
        return None


def get_center_of_mass(shapes):
    all_points = []
    for shape in shapes:
        all_points.append(shape.points)
    points_vars = torch.vstack(all_points)
    center = points_vars.mean(dim=0)

    return center, all_points


def get_deltas(all_points, center, device):
    
    deltas_from_center = []
    for points in all_points:
        deltas = (points - center).to(device)
        deltas_from_center.append(deltas)
    return deltas_from_center




