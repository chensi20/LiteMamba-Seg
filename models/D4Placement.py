import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.MambaSeg import ConvBlock, UnifiedBottleneck, unpack_tuple


class D4Placement_UNet(nn.Module):
    """
    Controlled D4 Mamba placement experiment.

    d4_mamba_pos:
        "none"        : No additional D4 Mamba
        "pre_up"      : Before D4 upsampling
        "pre_fusion"  : After upsampling, before skip fusion
        "post_fusion" : After skip fusion/projection, before decoder convolution
        "post_conv"   : After decoder convolution (corresponding to final D4 concept)

    Important:
    All four Mamba placement variants use exactly the same:
        - ResNet34 encoder
        - bottleneck
        - 512->256 pre-projection
        - D4 upsampling
        - skip fusion
        - 512->256 fusion projection
        - D4 ConvBlock
        - one identical 256-d Mamba block
        - remaining decoder

    Only the execution position of the D4 Mamba block changes.

    pre_fusion / post_fusion / post_conv:
        Mamba input = 256 x 22 x 22 for a 352x352 image.

    pre_up:
        Mamba input = 256 x 11 x 11.
        It is parameter-matched but naturally has lower FLOPs because
        the sequence length is shorter.
    """

    VALID_POSITIONS = {
        "none",
        "pre_up",
        "pre_fusion",
        "post_fusion",
        "post_conv",
    }

    def __init__(
        self,
        n_classes=1,
        d4_mamba_pos="post_conv",
        use_layer3_mamba=False,
        use_bottleneck_mamba=True,
        use_conv_refine=True,
        pretrained=True,
    ):
        super().__init__()

        if d4_mamba_pos not in self.VALID_POSITIONS:
            raise ValueError(
                f"Invalid d4_mamba_pos='{d4_mamba_pos}'. "
                f"Choose from {sorted(self.VALID_POSITIONS)}"
            )

        self.d4_mamba_pos = d4_mamba_pos
        self.use_layer3_mamba = use_layer3_mamba
        self.use_bottleneck_mamba = use_bottleneck_mamba
        self.use_conv_refine = use_conv_refine

        print("=" * 60)
        print("Building Controlled D4 Placement Model...")
        print(f"[*] D4 Mamba Position   : {d4_mamba_pos}")
        print(f"[*] Layer3 Mamba        : {use_layer3_mamba}")
        print(f"[*] Bottleneck Mamba    : {use_bottleneck_mamba}")
        print(f"[*] Conv Refinement     : {use_conv_refine}")
        print(f"[*] ImageNet Pretrained : {pretrained}")
        print("=" * 60)

        # ============================================================
        # Encoder: same ResNet34 structure as current MambaSeg.py
        # ============================================================
        weights = (
            models.ResNet34_Weights.DEFAULT
            if pretrained
            else None
        )

        resnet = models.resnet34(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.pool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # ============================================================
        # Optional Layer3 Mamba
        # ============================================================
        if self.use_layer3_mamba:
            self.layer3_mamba = UnifiedBottleneck(
                dim=256,
                use_mamba=True,
                use_conv_refine=use_conv_refine,
            )

        # ============================================================
        # Bottleneck: same logic as current MambaSeg.py
        # e5: [B, 512, 11, 11]
        # ============================================================
        self.bottleneck = UnifiedBottleneck(
            dim=512,
            use_mamba=use_bottleneck_mamba,
            use_conv_refine=use_conv_refine,
        )

        # ============================================================
        # Controlled D4 stage
        # ============================================================

        # Always executed for every placement variant.
        # [B,512,11,11] -> [B,256,11,11]
        #
        # This makes it possible for exactly the same d_model=256
        # Mamba block to operate at pre_up as well as D4 locations.
        self.d4_pre_proj = nn.Sequential(
            nn.Conv2d(
                512,
                256,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Always executed.
        # [B,256,11,11] -> [B,256,22,22]
        self.up4 = nn.ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2,
        )

        # After concatenation:
        # 256 decoder + 256 encoder = 512 channels.
        #
        # Always executed for every placement variant.
        self.d4_fusion_proj = nn.Sequential(
            nn.Conv2d(
                512,
                256,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Always executed.
        self.dec4 = ConvBlock(
            256,
            256,
        )

        # Every Mamba placement variant contains exactly ONE
        # identical D4 Mamba block.
        #
        # "none" intentionally does not contain this block and is
        # therefore the non-Mamba reference rather than a
        # parameter-matched placement variant.
        if self.d4_mamba_pos != "none":
            self.d4_mamba = UnifiedBottleneck(
                dim=256,
                use_mamba=True,
                use_conv_refine=use_conv_refine,
            )

        # ============================================================
        # Remaining decoder: same as current MambaSeg.py
        # ============================================================
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2,
        )
        self.dec3 = ConvBlock(
            128 + 128,
            128,
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2,
        )
        self.dec2 = ConvBlock(
            64 + 64,
            64,
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2,
        )
        self.dec1 = ConvBlock(
            32 + 64,
            32,
        )

        # ============================================================
        # Deep-supervision heads
        # ============================================================
        self.head1 = nn.Conv2d(
            32,
            n_classes,
            1,
        )
        self.head2 = nn.Conv2d(
            64,
            n_classes,
            1,
        )
        self.head3 = nn.Conv2d(
            128,
            n_classes,
            1,
        )
        self.head4 = nn.Conv2d(
            256,
            n_classes,
            1,
        )

    def forward(self, x):
        x = unpack_tuple(x)

        # ============================================================
        # Encoder
        # ============================================================
        e1 = self.stem(x)
        # 64 x 176 x 176

        e_pool = self.pool(e1)

        e2 = self.layer1(e_pool)
        # 64 x 88 x 88

        e3 = self.layer2(e2)
        # 128 x 44 x 44

        e4 = self.layer3(e3)
        # 256 x 22 x 22

        if self.use_layer3_mamba:
            e4 = self.layer3_mamba(e4)

        e5 = self.layer4(e4)
        # 512 x 11 x 11

        b = self.bottleneck(e5)
        # 512 x 11 x 11

        # ============================================================
        # Controlled D4 placement
        # ============================================================

        # Fixed projection:
        # 512 x 11 x 11 -> 256 x 11 x 11
        d4 = self.d4_pre_proj(b)

        # ------------------------------------------------------------
        # P1: Before upsampling
        # ------------------------------------------------------------
        if self.d4_mamba_pos == "pre_up":
            d4 = self.d4_mamba(d4)

        # Fixed upsampling:
        # 256 x 11 x 11 -> 256 x 22 x 22
        d4 = self.up4(d4)

        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(
                d4,
                size=e4.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # ------------------------------------------------------------
        # P2: After upsampling, before skip fusion
        # ------------------------------------------------------------
        if self.d4_mamba_pos == "pre_fusion":
            d4 = self.d4_mamba(d4)

        # Skip fusion
        d4 = torch.cat(
            [d4, e4],
            dim=1,
        )
        # 512 x 22 x 22

        # Fixed dimensionality projection
        d4 = self.d4_fusion_proj(d4)
        # 256 x 22 x 22

        # ------------------------------------------------------------
        # P3: After skip fusion, before decoder convolution
        # ------------------------------------------------------------
        if self.d4_mamba_pos == "post_fusion":
            d4 = self.d4_mamba(d4)

        # Fixed local convolutional reconstruction
        d4 = self.dec4(d4)
        # 256 x 22 x 22

        # ------------------------------------------------------------
        # P4: After decoder convolution
        # ------------------------------------------------------------
        if self.d4_mamba_pos == "post_conv":
            d4 = self.d4_mamba(d4)

        # ============================================================
        # Remaining decoder
        # ============================================================
        d3 = self.up3(d4)

        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(
                d3,
                size=e3.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d3 = torch.cat(
            [d3, e3],
            dim=1,
        )
        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(
                d2,
                size=e2.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d2 = torch.cat(
            [d2, e2],
            dim=1,
        )
        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1,
                size=e1.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        d1 = torch.cat(
            [d1, e1],
            dim=1,
        )
        d1 = self.dec1(d1)

        # ============================================================
        # Outputs
        # ============================================================
        out1 = self.head1(d1)
        out1 = F.interpolate(
            out1,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        if self.training:
            out2 = F.interpolate(
                self.head2(d2),
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            out3 = F.interpolate(
                self.head3(d3),
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            out4 = F.interpolate(
                self.head4(d4),
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

            return out1, out2, out3, out4

        return out1