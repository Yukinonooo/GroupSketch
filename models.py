import math
import os
import torch
import torch.nn as nn
from dino.dino_all import extract_video_features, VideoSVGFusion

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(self.pe[:x.size(0), :])

class SVGTimeEmbedding(nn.Module):
    """
    specifically designed for vector animation
    """
    def __init__(self, model_dim, max_period=128):
        super().__init__()
        self.max_period = max_period
        self.model_dim = model_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.SiLU(),
            nn.Linear(model_dim * 2, model_dim)
        )
        
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def get_timestep_embedding(self, timesteps, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=timesteps.device) / half)
        args = timesteps[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    def forward(self, data, num_frames, num_points):
        timesteps = torch.arange(num_frames, device=data.device, dtype=torch.float32)
        time_embeddings = self.get_timestep_embedding(timesteps, self.model_dim)
        time_embeddings = time_embeddings.unsqueeze(1)
        time_embeddings = time_embeddings.expand(-1, num_points, -1)
        time_embeddings = time_embeddings.reshape(1, num_frames * num_points, self.model_dim)
        processed_time_embed = self.time_embed(time_embeddings)
        processed_time_embed = self.layer_norm(processed_time_embed)
        
        return processed_time_embed
    
    
class PointModel(nn.Module):

    def __init__(self, input_dim, inter_dim, num_points_per_frame, num_frames,
                device, predict_global_frame_deltas, predict_only_global, 
                rotation_weight=1e-2, scale_weight=5e-2, shear_weight=5e-2, translation_weight=1,
                output_folder=None, dino_fusion=False, timeE=False):

        super().__init__()
        self.num_points_per_frame = num_points_per_frame
        self.num_frames = num_frames
        self.inter_dim = inter_dim
        self.input_dim = input_dim
        self.embed_dim = inter_dim
        self.predict_global_frame_deltas = predict_global_frame_deltas
        self.predict_only_global = predict_only_global
        self.dino_fusion = dino_fusion
        self.timeE = timeE
        self.project_points = nn.Sequential(nn.Linear(2, inter_dim),
                                            nn.LayerNorm(inter_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(inter_dim, inter_dim))

                
        if self.dino_fusion: 
            frame_paths = [os.path.join(output_folder, f"init_png/frame-{i}.png") for i in range(1, self.num_frames+1)]
            self.video_features = extract_video_features(frame_paths, batch_size=4)
            # initialize fusion module
            self.cross_fusion = VideoSVGFusion(
                dino_dim=self.video_features.shape[-1],  # DINOv2 feature dimension
                svg_dim=128,                        # SVG feature dimension
                num_heads=8,
                num_layers=3
            )

        self.embedding = nn.Embedding(input_dim, self.embed_dim)
        self.pos_encoder = PositionalEncoding(d_model=self.embed_dim, max_len=input_dim)
        # time embedding module
        if timeE:
            self.time_embedding = SVGTimeEmbedding(model_dim=self.embed_dim)
            
        self.inds = torch.tensor(range(int(input_dim / 2))).to(device)

        if predict_global_frame_deltas:
            self.rotation_weight = rotation_weight
            self.scale_weight = scale_weight
            self.shear_weight = shear_weight
            self.translation_weight = translation_weight

            self.frames_rigid_shared = nn.Sequential(nn.Flatten(),
                                            nn.Linear(int(input_dim * inter_dim / 2), inter_dim),
                                            nn.LayerNorm(inter_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(inter_dim, inter_dim),
                                            nn.LayerNorm(inter_dim),
                                            nn.LeakyReLU())

            self.frames_rigid_translation = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                        nn.LayerNorm(inter_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(inter_dim, inter_dim),
                                                        nn.LayerNorm(inter_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(inter_dim, inter_dim),
                                                        nn.LayerNorm(inter_dim),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(inter_dim, self.num_frames * 2))

            self.frames_rigid_rotation = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 1))
            
            self.frames_rigid_shear = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 2))

            self.frames_rigid_scale = nn.Sequential(nn.Linear(inter_dim, inter_dim),
                                                    nn.LayerNorm(inter_dim),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(inter_dim, self.num_frames * 2))
            #composer
            self.global_layers = nn.ModuleList([self.frames_rigid_shared, 
                                self.frames_rigid_translation, 
                                self.frames_rigid_rotation, 
                                self.frames_rigid_shear, 
                                self.frames_rigid_scale,
                                ])


    def get_position_encoding_representation(self, init_points):
        emb_xy = self.project_points(init_points) 
        embed = self.embedding(self.inds) * math.sqrt(self.embed_dim) 
        pos = self.pos_encoder(embed.unsqueeze(1)).permute(1, 0, 2)
        init_points_pos_enc = emb_xy + pos 
        return init_points_pos_enc

    def get_frame_deltas(self, init_points, init_points_pos_enc):
        # calculate global transform matrix for each frame, used to generate point offsets
        # input:
        # init_points: initial point coordinates [1, N*K, 2], N is number of points per frame, K is number of frames
        # init_points_pos_enc: position encoding of point coordinates [1, N*K, 128]
        # return:
        # frame_deltas: offset of each point [1, N*K, 2]
        frame_deltas = None
        #init_points_pos_enc.shape
        if self.predict_global_frame_deltas:
            # get features from shared MLP
            shared_params = self.frames_rigid_shared(init_points_pos_enc)

            dx, dy = self.frames_rigid_translation(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            dx = dx * self.translation_weight
            dy = dy * self.translation_weight

            theta = self.frames_rigid_rotation(shared_params).reshape(self.num_frames, 1) * self.rotation_weight
            cos_theta = torch.cos(theta) 
            sin_theta = torch.sin(theta)  

            shear_x, shear_y = self.frames_rigid_shear(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            shear_x = shear_x * self.shear_weight  
            shear_y = shear_y * self.shear_weight

            scale_x, scale_y = self.frames_rigid_shear(shared_params).reshape(self.num_frames, 2).chunk(2, axis=-1)
            scale_x = torch.ones_like(dx) + scale_x * self.scale_weight  
            scale_y = torch.ones_like(dy) + scale_y * self.scale_weight

            l1 = torch.concat([scale_x * (cos_theta - sin_theta * shear_x), scale_y * (cos_theta * shear_y - sin_theta), dx], axis=-1)

            l2 = torch.concat([scale_x * (sin_theta + cos_theta * shear_x), scale_y * (sin_theta * shear_y + cos_theta), dy], axis=-1)

            l3 = torch.concat([torch.zeros_like(dx), torch.zeros_like(dx), torch.ones_like(dx)], axis=-1)

            transform_mat = torch.stack([l1, l2, l3], axis=1)

            # repeat transform matrix to apply to all points in each frame
            transform_mat = torch.repeat_interleave(transform_mat, self.num_points_per_frame, dim=0)

            points_with_z = torch.concat([init_points, torch.ones_like(init_points)[:,:,0:1]], axis=-1)
            points_with_z = points_with_z.reshape(-1, 3, 1)

            # apply transform matrix to all points in each frame
            transformed_points = torch.matmul(transform_mat, points_with_z)[:, 0:2, :].reshape(1, -1, 2)
            frame_deltas = transformed_points - init_points

        return frame_deltas

    def forward(self, init_points):
        raise NotImplementedError("PointModel is an abstract class. Please inherit from it and implement a forward function.")

    def get_shared_params(self):
        """Get shared parameters from the model

        Returns:
            list: A list containing the following shared parameters:
                - project_points_p: Parameters of the point projection layer
                - embedding_p: Parameters of the embedding layer
                - pos_encoder_p: Parameters of the positional encoder
        """
        # Get parameters of the point projection layer
        project_points_p = list(self.project_points.parameters())
        # Get parameters of the embedding layer
        embedding_p = list(self.embedding.parameters())
        # Get parameters of the positional encoder
        pos_encoder_p = list(self.pos_encoder.parameters())
        basePara = project_points_p + embedding_p + pos_encoder_p
        # Get parameters of the time embedding layer
        if self.timeE:
            time_embedding_p = list(self.time_embedding.parameters())
            basePara = basePara + time_embedding_p
        # Get parameters of the cross fusion layer
        if self.dino_fusion:
            cross_fusion_p = list(self.cross_fusion.parameters())
            basePara = basePara + cross_fusion_p

        return basePara
        
    def get_points_params(self):
        # Get local path parameters from the model
        shared_params = self.get_shared_params()
        project_xy_p = list(self.project_xy.parameters())
        model_p = list(self.model.parameters())
        last_lin = list(self.last_linear_layer.parameters())
        return shared_params + project_xy_p + model_p + last_lin
        
    def get_global_params(self):
        """Get global path parameters from the model
        
        Returns:
            list: A list containing the following parameters:
                - shared_params: shared parameters from get_shared_params()
                - delta_p: parameters of the global transformation layer (global_layers)
        """
        # Get shared parameters (parameters of the point projection layer, embedding layer, and positional encoder)
        shared_params = self.get_shared_params()
        # Get parameters of the global transformation layer
        delta_p = list(self.global_layers.parameters())
        return shared_params + delta_p
    

class PointMLP(PointModel):
    def __init__(self, input_dim, inter_dim, num_points_per_frame, num_frames,
                device, predict_global_frame_deltas, predict_only_global,
                rotation_weight=1e-2, scale_weight=5e-2, shear_weight=5e-2, translation_weight=1,
                output_folder=None, dino_fusion=False, timeE=False):
        super().__init__(input_dim, inter_dim, num_points_per_frame, num_frames,
                        device, predict_global_frame_deltas, predict_only_global,
                        rotation_weight, scale_weight, shear_weight, translation_weight,
                        output_folder, dino_fusion, timeE)
        self.num_points = num_points_per_frame
        self.num_frames = num_frames
        # projection layer: map input point coordinates to intermediate dimension space
        self.project_xy = nn.Sequential(nn.Flatten(),
                                        nn.Linear(int(input_dim * inter_dim / 2), inter_dim),
                                        nn.LayerNorm(inter_dim),
                                        nn.LeakyReLU())

        self.model = nn.Sequential(
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
            nn.Linear(inter_dim, inter_dim),
            nn.LayerNorm(inter_dim),
            nn.LeakyReLU(),
        )

        # output layer: map features back to coordinate space
        self.last_linear_layer = nn.Linear(inter_dim, input_dim)

    def forward(self, init_points):
        """
        Forward propagation function
        Args:
            init_points: Initial point coordinates [1, N*K, 2]
        Returns:
            delta_xy: Predicted point offsets [N*K, 2]
        """
        init_points_pos_enc = self.get_position_encoding_representation(init_points)
        init_points_pos_enc_local = init_points_pos_enc
        if self.timeE:
            init_points_pos_enc = self.time_embedding(init_points_pos_enc, self.num_frames, self.num_points)
        if self.dino_fusion:           
            # cross modal fusion
            init_points_pos_enc = self.cross_fusion(init_points_pos_enc, self.video_features)
           
        frame_deltas = self.get_frame_deltas(init_points, init_points_pos_enc)

        if self.predict_only_global:
            return (frame_deltas).squeeze(0)

        project_xy = self.project_xy(init_points_pos_enc_local)  # Flatten, output is [1, 128]
        delta = self.model(project_xy)  # [1,128]
        delta_xy = self.last_linear_layer(delta).reshape(init_points.shape)  # [1,128] -> [1, N*K, 2]

        # add global and local transform offsets
        if self.predict_global_frame_deltas:
            delta_xy = delta_xy + frame_deltas
        return delta_xy.squeeze(0)