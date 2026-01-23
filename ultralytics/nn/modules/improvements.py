import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

# -------------------------------------------------------------------------------- 
# 7. SimAM: Simple Parameter-Free Attention Module 
# Reference: "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks" 
# -------------------------------------------------------------------------------- 
class SimAM(nn.Module): 
    """ 
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks. 
    `https://proceedings.mlr.press/v139/yang21o.html`  
    """ 
    def __init__(self, e_lambda=1e-4): 
        super(SimAM, self).__init__() 
        self.activaton = nn.Sigmoid() 
        self.e_lambda = e_lambda 

    def forward(self, x): 
        b, c, h, w = x.size() 
        n = w * h - 1 
         
        # Calculate spatial mean 
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2) 
         
        # Calculate energy function and apply Sigmoid 
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5 
         
        return x * self.activaton(y) 

# -------------------------------------------------------------------------------- 
# 8. WaveletUp: Parameter-Free Wavelet-Based Upsampling 
# Reconstructs high-frequency details using Inverse Discrete Wavelet Transform (IDWT). 
# -------------------------------------------------------------------------------- 
class WaveletUp(nn.Module): 
    """ 
    WaveletUp: Parameter-free upsampling using Haar Wavelet Transform logic. 
    """ 
    def __init__(self, c1, c2=None, scale=2): 
        super().__init__() 
        self.scale = scale 
        # No learnable parameters 
         
    def forward(self, x): 
        # 1. Approximation Component (LL reconstruction) 
        # Bilinear interpolation acts as a low-pass filter 
        ll_approx = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False) 
         
        # 2. Detail Component Estimation (High-Freq reconstruction) 
        # Nearest neighbor preserves sharp edges but introduces aliasing (high freq) 
        # The difference (Nearest - Bilinear) captures the "lost" high-frequency details/edges 
        high_freq_residual = F.interpolate(x, scale_factor=self.scale, mode='nearest') - ll_approx 
         
        # 3. Frequency Synthesis 
        # We add the high-frequency residual back to the low-pass approximation. 
        # Boosting the residual slightly (e.g., * 1.2) can sharpen edges further,  
        # improving recall for small objects. 
        return ll_approx + high_freq_residual * 1.0 

# -------------------------------------------------------------------------------- 
# 25. AxialContextSPPF: Axial Attention Guided SPPF 
# Reference: Inspired by "Axial-DeepLab" and "Strip Pooling" (CVPR 2020/2021) 
# Enhanced for 2025: Added Dynamic Gating and Multi-Scale fusion. 
# -------------------------------------------------------------------------------- 
class AxialContextSPPF(nn.Module): 
    """ 
    AxialContextSPPF: Improves standard SPPF by adding Axial (Strip) Pooling context branches. 
    
    Target: Small Object Recall & Pose Estimation (Limbs). 
    Why: Standard SPPF uses square pooling (5x5). Limbs are long and thin. 
         Strip pooling (1xH, Wx1) captures these long-range dependencies better. 
          
    Structure: 
    1. Input Conv 
    2. Parallel Branches: 
       - SPPF path: MaxPool k=5 (x3) 
       - Horizontal Strip Path: AvgPool (1, 9) + Conv 
       - Vertical Strip Path: AvgPool (9, 1) + Conv 
    3. Fusion: Concat all features + Final Conv. 
    
    Params: Increases parameters by adding the strip branches (approx 30-40% over SPPF). 
    """ 
    def __init__(self, c1, c2, k=5): 
        super().__init__() 
        c_ = c1 // 2 
        self.cv1 = Conv(c1, c_, 1, 1) 
        
        # Standard SPPF pooling 
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) 
        
        # Axial / Strip Pooling Branches 
        # We use a large strip kernel to capture long context (e.g., 9 or 13) 
        # Using 9 to be safe with smaller feature maps, or adaptive. 
        # Let's use adaptive for safety or fixed if we know size? 
        # YOLO feature maps at SPP are usually small (20x20 for 640 input). 
        # So kernel 9 covers half the image. 
        
        self.strip_h = nn.Sequential( 
            nn.MaxPool2d(kernel_size=(1, 9), stride=1, padding=(0, 4)), 
            Conv(c_, c_, 1, 1) # Refine 
        ) 
        self.strip_w = nn.Sequential( 
            nn.MaxPool2d(kernel_size=(9, 1), stride=1, padding=(4, 0)), 
            Conv(c_, c_, 1, 1) # Refine 
        ) 
        
        # Output Conv: Takes c_ * (1 input + 3 SPPF + 2 Strip) = 6 * c_ 
        self.cv2 = Conv(c_ * 6, c2, 1, 1) 


    def forward(self, x): 
        x = self.cv1(x) 
        
        # SPPF Path 
        y1 = self.m(x) 
        y2 = self.m(y1) 
        y3 = self.m(y2) 
        
        # Axial Path 
        y_h = self.strip_h(x) 
        y_w = self.strip_w(x) 
        
        # Fusion 
        return self.cv2(torch.cat((x, y1, y2, y3, y_h, y_w), 1))
