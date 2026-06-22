import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from mamba_ssm import Mamba 

def unpack_tuple(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(True)
        )
    def forward(self, x): 
        return self.conv(x)


class UnifiedBottleneck(nn.Module):
    def __init__(self, dim, use_mamba=True, use_conv_refine=True):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_conv_refine = use_conv_refine
        
        if self.use_mamba:
            self.mamba = Mamba(
                d_model=dim, 
                d_state=16, 
                d_conv=4, 
                expand=2
            )
            self.norm = nn.LayerNorm(dim)
            
        if self.use_conv_refine:
            self.conv_refine = nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(True)
            )

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        out = x
       
        if self.use_mamba:
            x_flat = out.flatten(2).permute(0, 2, 1)
            x_flat = self.norm(x_flat)
            out = self.mamba(x_flat)
            out = out.permute(0, 2, 1).view(B, C, H, W)
            
        if self.use_conv_refine:
            out = self.conv_refine(out)
            
        if not self.use_mamba and not self.use_conv_refine:
            return out
            
        return out + residual


class MambaSeg_UNet(nn.Module):
 
    def __init__(self, n_classes=1, use_layer3_mamba=False, use_d4_mamba=False, use_bottleneck_mamba=False, use_conv_refine=False): 
        super().__init__()
        
        self.use_layer3_mamba = use_layer3_mamba 
        self.use_d4_mamba = use_d4_mamba        
        self.use_bottleneck_mamba = use_bottleneck_mamba 
        self.use_conv_refine = use_conv_refine 
        
        print("="*50)
        print(f"Building Model Variants (Full Ablation Base)...")
        print(f"[*] ResNet34 Encoder   : True")
        print(f"[*] Layer3 Mamba       : {use_layer3_mamba}")
        print(f"[*] Decoder D4 Mamba   : {use_d4_mamba}")     
        print(f"[*] Bottleneck Mamba   : {use_bottleneck_mamba}")
        print(f"[*] Conv Refinement    : {use_conv_refine}")
        print("="*50)
        
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  
        
        if self.use_layer3_mamba:
            self.layer3_mamba = UnifiedBottleneck(
                dim=256, 
                use_mamba=True, 
                use_conv_refine=use_conv_refine 
            )
            
        self.bottleneck = UnifiedBottleneck(
            dim=512, 
            use_mamba=use_bottleneck_mamba, 
            use_conv_refine=use_conv_refine
        )
        
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(256 + 256, 256)
        
    
        if self.use_d4_mamba:
            self.d4_mamba = UnifiedBottleneck(
                dim=256, 
                use_mamba=True, 
                use_conv_refine=use_conv_refine 
            )
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(32 + 64, 32)
        
        self.head1 = nn.Conv2d(32, n_classes, 1)
        self.head2 = nn.Conv2d(64, n_classes, 1)
        self.head3 = nn.Conv2d(128, n_classes, 1)
        self.head4 = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        x = unpack_tuple(x)
        
        e1 = self.stem(x)       
        e_pool = self.pool(e1)  
        e2 = self.layer1(e_pool)  
        e3 = self.layer2(e2)      
        
        e4 = self.layer3(e3)  
        if self.use_layer3_mamba:
            e4 = self.layer3_mamba(e4) 
            
        e5 = self.layer4(e4)  
        b = self.bottleneck(e5) 
        
        d4 = self.up4(b)
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
      
        if self.use_d4_mamba:
            d4 = self.d4_mamba(d4)
        
        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)              
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out1 = self.head1(d1)
        out1 = F.interpolate(out1, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        if self.training:
            out2 = F.interpolate(self.head2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
            out3 = F.interpolate(self.head3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
            out4 = F.interpolate(self.head4(d4), size=x.shape[2:], mode='bilinear', align_corners=True)
            return out1, out2, out3, out4
            
        return out1