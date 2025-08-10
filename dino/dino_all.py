import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

def init_dinov2_model(name: str = 'dinov2_vitl14_reg'):
    dinov2_model = torch.hub.load('/home/liangguotao/.cache/torch/hub/facebookresearch_dinov2_main', name, source='local')# change to your own dinov2 path
    dinov2_model.to(torch.device('cuda:0'))
    dinov2_model.eval()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return dinov2_model, transform

def extract_video_features(frame_paths, batch_size=4):
    """Batch extract DINOv2 features from video frames
    
    Args:
        frame_paths: list of video frame paths
        batch_size: number of images to process in each batch, to reduce memory usage
        
    Returns:
        video features tensor [num_frames, num_patches, feature_dim]
    """
    dinov2_model, transform = init_dinov2_model()
    
    all_features = []
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            image = Image.open(path)
            image = transform(image)
            batch_images.append(image)
        
        batch_tensor = torch.stack(batch_images, dim=0).to('cuda:0')
        
        with torch.no_grad():
            batch_features = dinov2_model(batch_tensor, is_training=True)['x_prenorm']
        
        batch_patchtokens = F.layer_norm(batch_features, batch_features.shape[-1:], eps=1e-6)
        
        for j in range(batch_patchtokens.size(0)):
            all_features.append(batch_patchtokens[j])
    
    # stack all image features [num_frames, num_patches, feature_dim]
    stacked_features = torch.stack(all_features, dim=0)
    
    # free up memory
    del dinov2_model
    torch.cuda.empty_cache()
    
    return stacked_features.detach()

class TemporalAttention(nn.Module):
    """Temporal attention, process inter-frame relationships"""
    def __init__(self, feature_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x: [F, P, C] - F is the number of frames, P is the number of patches per frame, C is the feature dimension
        Returns:
            time-fused features: [P, C]
        """
        F, P, C = x.shape
        x_norm = self.norm(x)
        # 调整形状以便进行时间注意力 [P, F, C]
        x_reshaped = x_norm.permute(1, 0, 2)
        
        # 对每个位置执行时间自注意力
        temporal_out, _ = self.temporal_attn(
            query=x_reshaped,
            key=x_reshaped,
            value=x_reshaped
        )
        
        # 将结果平均池化，获得每个位置的时间融合特征 [P, C]
        return temporal_out.mean(dim=1)

class SpatioTemporalCrossAttention(nn.Module):
    """Spatio-temporal cross-attention module"""
    def __init__(self, feature_dim=128, num_heads=8):
        super().__init__()
        self.norm_svg = nn.LayerNorm(feature_dim)
        self.norm_img = nn.LayerNorm(feature_dim)
        
        # temporal attention to process video frame sequence
        self.temporal_attn = TemporalAttention(feature_dim)
        
        # spatial cross-attention: SVG point features (query) with time-fused image features (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.norm_final = nn.LayerNorm(feature_dim)
        
    def forward(self, svg_features, video_features):
        """
        Args:
            svg_features: [B, N, C] - SVG point features
            video_features: [F, P, C] - F frames, P patches per frame, C-dimensional features
        Returns:
            enhanced features
        """
        svg_norm = self.norm_svg(svg_features)
        video_norm = self.norm_img(video_features)
        
        # time fusion, get time-aware features
        temporal_features = self.temporal_attn(video_norm)  # [P, C]
        # spatial cross-attention
        B = svg_features.size(0)
        attn_out, _ = self.cross_attn(
            query=svg_norm,
            key=temporal_features.unsqueeze(0).expand(B, -1, -1),
            value=temporal_features.unsqueeze(0).expand(B, -1, -1)
        )
        
        # residual connection
        fused_features = svg_features + attn_out
        
        # FFN
        enhanced_features = fused_features + self.ffn(self.norm_final(fused_features))
        
        return enhanced_features

class VideoSVGFusion(nn.Module):
    """Spatio-temporal fusion module for video frames and SVG point features"""
    def __init__(self, dino_dim=1024, svg_dim=128, num_heads=8, num_layers=3):
        super().__init__()
        # project DINOv2 features to SVG feature space
        self.proj = nn.Linear(dino_dim, svg_dim)
        
        # multi-layer spatio-temporal cross-attention
        self.layers = nn.ModuleList([
            SpatioTemporalCrossAttention(svg_dim, num_heads) 
            for _ in range(num_layers)
        ])
    
    def forward(self, svg_features, dino_features):
        """
        Args:
            svg_features: [B, N, C_svg] - SVG point features
            dino_features: [F, B*P, C_dino] - DINOv2 video features (F frames)
        Returns:
            enhanced_svg: [B, N, C_svg] - Enhanced SVG features
        """
        # project DINOv2 features to SVG feature space
        projected_features = self.proj(dino_features)  # [F, B*P, C_svg]
        
        # through multi-layer spatio-temporal attention
        enhanced_svg = svg_features
        for layer in self.layers:
            enhanced_svg = layer(enhanced_svg, projected_features)
            
        return enhanced_svg
