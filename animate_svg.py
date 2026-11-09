from painter import Painter, PainterOptimizer
from losses import SDSVideoLoss
import utils
import os
import matplotlib.pyplot as plt
import torch
import pydiffvg
from tqdm import tqdm
from pytorch_lightning import seed_everything
import argparse
import wandb
import numpy as np
from torchvision import transforms
import torchvision
import copy
import time  

model_choice = {
    "damo-vilab/text-to-video-ms-1.7b": SDSVideoLoss
}
def parse_arguments():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--target", type=str, default="afterMove/player_nocolor/input", help="file name of the svg to be animated in afterMove folder generally")
    parser.add_argument("--caption", type=str, default="", help="Prompt for animation. verify first that this prompt works with the original text2vid model. If left empty will try to find prompt in utils.py")
    parser.add_argument("--output_folder", type=str, default="horse_256", help="folder name to save the results")
    parser.add_argument("--seed", type=int, default=1000)

    # Diffusion related & Losses
    parser.add_argument("--model_name", type=str, default="damo-vilab/text-to-video-ms-1.7b")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--render_size_h", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--render_size_w", type=int, default=256, help="should fit the default settings of the chosen video model (under 'model_name')")
    parser.add_argument("--num_frames", type=int, default=24, help="should fit the default settings of the chosen video model (under 'model_name')")
    
    # SDS related
    parser.add_argument("--sds_timestep_low", type=int, default=50) 
    parser.add_argument("--same_noise_for_frames", action="store_true", help="sample noise for one frame and repeat across all frames")
    parser.add_argument("-augment_frames", type=bool, default=True, help="whether to randomely augment the frames to prevent adversarial results")
    
    # Memory saving related
    parser.add_argument("--use_xformers", action="store_true", help="Enable xformers for unet")
    parser.add_argument("--del_text_encoders", action="store_true", help="delete text encoder and tokenizer after encoding the prompts")

    # Optimization related
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--optim_points", action='store_false', help="whether to optimize the points (x,y) of the object or not.defaul is True")
    parser.add_argument("--optim_points_with_mlp", type=bool, default=True, help="whether to optimize the points with an MLP")

    parser.add_argument("--split_global_loss", action='store_false', help="whether to use a different loss for the center prediction.defaul is True")
    parser.add_argument("--guidance_scale_global", type=float, default=40, help="SDS guidance scale for the global path")
    parser.add_argument("--lr_base_global", type=float, default=0.0001, help="Base learning rate for the global path")

    # MLP architecture (points)
    parser.add_argument("--predict_global_frame_deltas", type=float, default=1, help="whether to predict a global delta per frame, the value is the weight of the output")
    parser.add_argument("-predict_only_global", action='store_true', help="whether to predict only global deltas")
    parser.add_argument("--inter_dim", type=int, default=128)
    parser.add_argument("--use_shared_backbone_for_global", action='store_true',
                        help="Whether to use the same backbone for the global prediction as for per point prediction")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--normalize_input", type=int, default=0)
    parser.add_argument("--translation_layer_norm_weight", type=int, default=0)

    parser.add_argument("--rotation_weight", type=float, default=0.01, help="Scale factor for global transform matrix 'rotation' terms")
    parser.add_argument("--scale_weight", type=float, default=0.05, help="Scale factor for global transform matrix 'scale' terms")
    parser.add_argument("--shear_weight", type=float, default=0.1, help="Scale factor for global transform matrix 'shear' terms")
    parser.add_argument("--translation_weight", type=float, default=0.01, help="Scale factor for global transform matrix 'translation' terms")
    # Learning rate related (can be simplified, taken from vectorFusion)
    parser.add_argument("--lr_local", type=float, default=0.005)
    parser.add_argument("--lr_init", type=float, default=0.002)
    parser.add_argument("--lr_final", type=float, default=0.0008)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    parser.add_argument("--lr_delay_steps", type=float, default=100)
    parser.add_argument("--const_lr", type=int, default=0)

    # Display related
    parser.add_argument("--display_iter", type=int, default=50)
    parser.add_argument("--save_vid_iter", type=int, default=100)

    # wandb
    parser.add_argument("-report_to_wandb", action='store_true')
    parser.add_argument("--wandb_user", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    parser.add_argument("--folder_as_wandb_run_name", action="store_true")

    # use Canvas to make motion data
    parser.add_argument("--use_canvas", default=True, help="Whether to use Canvas to make motion data")
    parser.add_argument("--affine_prior", type=str, default=None, help="Explicit per-group affine initialization YAML; replaces Canvas translations")
    # use dinov2 for image feature fusion
    parser.add_argument("--dino", action="store_true", help="Whether to use dinov2 for image feature fusion")
    parser.add_argument("--single_group_network", action="store_true", help="Use one GDN after applying unchanged per-path initialization priors")
    parser.add_argument('--merge_groups', action='store_true', help='Merge all SVGs into one group for processing')
    parser.add_argument("--timeE", action="store_true", help="Whether to use Time Encoding for image feature fusion")
    
    # Web interface related  
    parser.add_argument("--progress_callback_file", type=str, default=None, help="File path to write training progress for web interface")

    
    args = parser.parse_args()
    seed_everything(args.seed)
    target = os.path.join(args.target, "postprocess")
    if not args.caption:
        args.caption = utils.get_caption(target)
        
    print("=" * 50)
    print("target:", target)
    print("caption:", args.caption)
    print("=" * 50)

    if args.folder_as_wandb_run_name:
        args.wandb_run_name = args.output_folder

    args.output_folder = f"./output_videos/{args.output_folder}"
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(f"{args.output_folder}/svg_logs", exist_ok=True)
    os.makedirs(f"{args.output_folder}/mp4_logs", exist_ok=True)

    if args.report_to_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_user,
                    config=args, name=args.wandb_run_name, id=wandb.util.generate_id())

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    # save full command to command.txt
    import sys
    command_file = os.path.join(args.output_folder, "command.txt")
    with open(command_file, "w") as f:
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_devices:
            f.write(f"CUDA_VISIBLE_DEVICES={cuda_devices} ")
        f.write("python " + sys.argv[0])
        for arg in sys.argv[1:]:
            if " " in arg:  
                f.write(f' "{arg}"')
            else:
                f.write(f" {arg}")
        f.write("\n")

    return args


def plot_video_seq(x_aug, orig_aug, cfg, step):
    pair_concat = torch.cat([orig_aug.squeeze(0).detach().cpu(), x_aug.squeeze(0).detach().cpu()])
    grid_img = torchvision.utils.make_grid(pair_concat, nrow=cfg.num_frames)
    plt.figure(figsize=(30,10))
    plt.imshow(grid_img.permute(1, 2, 0), vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"frames_iter{step}")
    plt.tight_layout()
    if cfg.report_to_wandb:
        wandb.log({"frames": wandb.Image(plt)}, step=step)


def get_augmentations():
    """get data augmentation
    
    return:
        augment_trans: data augmentation, include:
            1. RandomPerspective - random perspective transform
               - fill=1: fill value is 1(white)
               - p=1.0: 100% execute this transform
               - distortion_scale=0.5: distortion scale is 0.5
               
            2. RandomResizedCrop - random crop and resize
               - size=(256,256): output size is 256x256
               - scale=(0.4,1): crop area ratio is between 0.4 and 1
               - ratio=(1.0,1.0): keep 1:1 aspect ratio
    """
    augemntations = []
    augemntations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augemntations.append(transforms.RandomResizedCrop(
        size=(256,256), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augemntations)
    return augment_trans

def remove_last_folder(path):
    """
    Remove the last folder or filename part from a path string.

    param path: Input path string
    return: Path string after removing the last component
    """
    if not path:
        return path  

    if path.endswith('/'):
        path = path[:-1]

    parts = path.rsplit('/', 1)
    if len(parts) > 1:
        return parts[0]  
    return path  

if __name__ == "__main__":
    # Parse command-line arguments
    cfg = parse_arguments()
    target = os.path.join(cfg.target, "postprocess")
    utils.svg_to_png_single(target, cfg.output_folder)
    os.makedirs(os.path.join(cfg.output_folder, "init_png"), exist_ok=True)
    folder = remove_last_folder(cfg.target)
    utils.svg_to_png(folder, os.path.join(cfg.output_folder, "init_png"))
    # Initialize Painter
    painter = Painter(cfg, target, num_frames=cfg.num_frames, device=cfg.device)
    composed_global_params = []
    composed_points_params = []

    # Initialize optimizer
    optimizer = PainterOptimizer(cfg)
    if cfg.predict_global_frame_deltas:
        composed_global_params = painter.get_global_params()
    if cfg.optim_points_with_mlp:
        composed_points_params = painter.get_points_params()
    optimizer.init_optimizers(composed_global_params, composed_points_params)
    
    # Get data augmentation transforms (transforms.Compose)
    data_augs = get_augmentations()

    motion_data = []
    # test if SVG and initial frames are loaded correctly
    frames_tensor, frames_svg, points_init_frame = painter.render_frames_to_tensor(mlp=True)
    output_vid_path = f"{cfg.output_folder}/init_vid.mp4"
    utils.save_mp4_from_tensor(frames_tensor, output_vid_path)
    # save them to frames folder
    frames, paths = utils.process_and_save_tensor_frames(
        frames_tensor, 
        save_dir=os.path.join(cfg.output_folder, "frames"),
        base_filename="my_frame"
    )
                    
    # Initialize SDS video loss
    SDSLoss = model_choice[cfg.model_name]
    sds_loss = SDSLoss(cfg, cfg.device)
    # if needed, set different parameters for the global path
    # reuse the same text-2-video diffusion pipeline
    if cfg.predict_global_frame_deltas and cfg.split_global_loss:
        global_cfg = copy.deepcopy(cfg)
        if cfg.guidance_scale_global is not None:
            global_cfg.guidance_scale = cfg.guidance_scale_global
            global_sds_loss = SDSLoss(global_cfg, global_cfg.device, reuse_pipe=True)

    # prepare original frames tensor
    orig_frames = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
    orig_frames = orig_frames.repeat(cfg.batch_size, 1, 1, 1, 1)
    
    sds_losses_and_opt_kwargs = []

    if cfg.predict_global_frame_deltas:
        # add two loss pairs:
        # - First pair: local deformation loss (skip_global=True → only optimize local)
        sds_losses_and_opt_kwargs.append((sds_loss, {"skip_global": True}))
        # - Second pair: global transform loss (skip_points=True → only optimize global)
        sds_losses_and_opt_kwargs.append((global_sds_loss, {"skip_points": True}))
    else:
        # add one loss pair, optimize both global and local
        sds_losses_and_opt_kwargs.append((sds_loss, {}))# empty dict means skip nothing
        
    t_range = tqdm(range(cfg.num_iter + 1))
    start_time = time.time()
    
    # Progress callback function
    def write_progress(step, total_steps, loss_val, elapsed_time, lr_points, lr_global, status="training"):
        if cfg.progress_callback_file:
            remaining_time = (elapsed_time / (step + 1)) * (total_steps - step - 1) if step > 0 else 0
            progress_data = {
                "step": step,
                "total_steps": total_steps,
                "progress": (step / total_steps) * 100,
                "loss": float(loss_val) if loss_val is not None else 0.0,
                "elapsed_time": int(elapsed_time),
                "estimated_remaining": int(remaining_time),
                "lr_points": float(lr_points) if lr_points else 0.0,
                "lr_global": float(lr_global) if lr_global else 0.0,
                "status": status
            }
            try:
                import json
                with open(cfg.progress_callback_file, 'w') as f:
                    json.dump(progress_data, f)
            except:
                pass  # Ignore errors to not affect training
    
    for step in t_range:
        iter_start_time = time.time()
        
        # Initialize loss for this step
        step_loss = None
        
        for i, (curr_sds_loss, opt_kwargs) in enumerate(sds_losses_and_opt_kwargs):
            is_last = (i == len(sds_losses_and_opt_kwargs) - 1)
            logs = {}
            optimizer.zero_grad_()

            # Render the frames (inc. network forward pass)
            vid_tensor, frames_svg, new_points = painter.render_frames_to_tensor() # (K, 256, 256, 3)
            x = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3)  # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
            x = x.repeat(cfg.batch_size, 1, 1, 1, 1)

            # Apply augmentations if needed
            if cfg.augment_frames:
                B, F, C, H, W = x.shape  # [1, 24, 3, 256, 256]
                x_reshaped = x.view(B*F, C, H, W)  # [24, 3, 256, 256]
                x_aug = data_augs(x_reshaped)
                x_aug = x_aug.view(B, F, C, H, W)  # restore original shape
            else:
                x_aug = x
                orig_frames_aug = orig_frames
            
            # Compute SDS loss. Note: The returned loss value is always a placeholder "1".
            # SDS is applied by changing the backprop calculation, see SpecifyGradient in losses.py 
            ref_first_image = x_aug[:,0,:,:,:]
            if SDSLoss == SDSVideoLoss:
                loss_sds = curr_sds_loss(x_aug)
            else:
                loss_sds = curr_sds_loss(x_aug, ref_first_image=ref_first_image, ref_img=ref_first_image)
                
            loss = loss_sds
            step_loss = loss.item()  # Store the loss value

            t_range.set_postfix({'loss': loss.item()})
        
            loss.backward()  
            
            optimizer.step_(**opt_kwargs)
            
            loss_suffix = "_global" if "skip_points" in opt_kwargs else ""
            logs.update({f"loss{loss_suffix}": loss.detach().item(), f"loss_sds{loss_suffix}": loss_sds.detach().item()})

            lr_points = optimizer.get_lr("points")
            lr_global = optimizer.get_lr("global")
            
            t_range.set_postfix({
                'lr_points': f'{lr_points:.2e}' if lr_points else 'None',
                'lr_global': f'{lr_global:.2e}' if lr_global else 'None'
            })

        if not cfg.const_lr:
            optimizer.update_lr()

        logs.update({"lr_points": optimizer.get_lr("points"), "step": step})
        
        iter_end_time = time.time()
        iter_time = iter_end_time - iter_start_time
        elapsed_time = iter_end_time - start_time 
        
        # Write progress callback
        write_progress(step, cfg.num_iter, step_loss, elapsed_time, lr_points, lr_global)
                     
        if cfg.report_to_wandb:
            wandb.log(logs, step=step)

        if step % cfg.save_vid_iter == 0:
            utils.save_mp4_from_tensor(vid_tensor, f"{cfg.output_folder}/mp4_logs/{step}.mp4")
            utils.save_vid_svg(frames_svg, f"{cfg.output_folder}/svg_logs", step, painter.canvas_width, painter.canvas_height)
            if cfg.report_to_wandb:
                video_to_save = vid_tensor.permute(0,3,1,2).detach().cpu().numpy()
                video_to_save = ((video_to_save / video_to_save.max()) * 255).astype(np.uint8)
                wandb.log({"video": wandb.Video(video_to_save, fps=8)}, step=step)
                plot_video_seq(x_aug, orig_frames_aug, cfg, step)
    total_time = time.time() - start_time
    with open(os.path.join(cfg.output_folder, "training_time_summary.txt"), "w") as f:
        f.write(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average iteration time: {total_time/cfg.num_iter:.2f} seconds\n")
        f.write(f"Number of iterations: {cfg.num_iter}\n")
    
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")            

    if cfg.report_to_wandb:
        wandb.finish()
    
    # Saves a high quality .gif from the final SVG frames
    utils.save_hq_video(cfg.output_folder, iter_=cfg.num_iter)
    
    # Write final completion status
    final_elapsed = time.time() - start_time
    final_lr_points = optimizer.get_lr("points") if 'optimizer' in locals() else 0.0
    final_lr_global = optimizer.get_lr("global") if 'optimizer' in locals() else 0.0
    write_progress(cfg.num_iter, cfg.num_iter, step_loss if 'step_loss' in locals() else 0.0, 
                   final_elapsed, final_lr_points, final_lr_global, "completed")

