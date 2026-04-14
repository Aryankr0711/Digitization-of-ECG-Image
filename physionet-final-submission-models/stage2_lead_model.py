"""
Stage2 Lead Model with Cross-Lead Feature Fusion

This module provides a segmentation model that processes 4 separate lead images
(cropped around zero_mv baselines) with cross-lead feature fusion to learn
inter-lead relationships.

Key features:
- Input: (B, 4, 3, 300, W) - 4 lead images, each 300px height (±150 around zero_mv)
- Output: (B, 4, 1, 300, W) - Single-channel mask per lead
- Shared encoder/decoder across all leads
- Multiple fusion strategies:
  * None: No fusion (baseline)
  * Conv2D: Per-lead channel reduction + concat + mix (recommended)
  * Shared Conv2D: Shared channel reduction (parameter efficient)
  * Conv3D: 3D convolution along lead dimension (NEW)
- Residual connections for all fusion types

Usage:
    # Conv2D fusion (recommended)
    model = Net(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        fusion_type='conv2d',
        fusion_levels=[1, 2, 3, 4]
    )

    # Conv3D fusion (NEW)
    model = Net(
        encoder_name='resnet34',
        fusion_type='conv3d',
        fusion_levels=[1, 2, 3, 4],
        conv3d_depth=2
    )

    # No fusion (baseline)
    model = Net(fusion_type='none')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation_models_pytorch is required. Install with: "
        "pip install segmentation-models-pytorch"
    )


class Conv3dBlock(nn.Module):
    """3D Convolution block for cross-lead feature fusion.

    Applies 3D convolution along (channel, lead, height, width) dimensions
    with batch normalization and LeakyReLU activation.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: 3D kernel size (lead_dim, H_dim, W_dim), default (3,3,3)
        padding: 3D padding (lead_dim, H_dim, W_dim), default (0,1,1)
        stride: 3D stride (lead_dim, H_dim, W_dim), default (1,1,1)
        padding_mode: Padding mode, default 'replicate'

    Example:
        >>> block = Conv3dBlock(64, 64, stride=(2,1,1))
        >>> x = torch.randn(2, 64, 4, 75, 544)  # (B, C, L, H, W)
        >>> out = block(x)  # (2, 64, 2, 75, 544) - lead dim reduced
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        padding=(0, 1, 1),
        stride=(1, 1, 1),
        padding_mode='replicate'
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, L, H, W) where L=lead dimension

        Returns:
            (B, C, L', H, W) where L' depends on stride
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class CrossLeadFusion(nn.Module):
    """Fuses features across 4 leads with residual connection.

    This module implements cross-lead feature fusion to enable the model to learn
    relationships between different ECG leads. The fusion process:

    1. Reshape (B*4, C, H, W) → (B, 4, C, H, W) to separate leads
    2. Per-lead channel reduction: Conv2d(C → C//4) for each lead independently
    3. Concatenate reduced features: 4 × C//4 = C channels
    4. Mix features: Conv2d(C → C) with two 3×3 conv blocks
    5. Broadcast mixed features: (B, C, H, W) → (B, 4, C, H, W)
    6. Residual connection: output = input + fused
    7. Reshape back: (B, 4, C, H, W) → (B*4, C, H, W)

    Args:
        channels: Number of feature channels at this encoder level (e.g., 64, 128, 256, 512)
        num_leads: Number of leads (default: 4)
        fusion_type: Fusion strategy - 'conv2d', 'shared_conv2d', or 'conv3d' (default: 'conv2d')
        reduction_ratio: Channel reduction ratio for conv2d types (default: 4, reduces C → C//4)
        conv3d_depth: Number of Conv3dBlocks for conv3d fusion (default: 2)
    """

    def __init__(
        self,
        channels,
        num_leads=4,
        fusion_type='conv2d',
        reduction_ratio=4,
        conv3d_depth=2,
    ):
        super().__init__()
        self.channels = channels
        self.num_leads = num_leads
        self.fusion_type = fusion_type

        # Validate fusion_type
        valid_types = ['conv2d', 'shared_conv2d', 'conv3d']
        if fusion_type not in valid_types:
            raise ValueError(
                f"fusion_type must be one of {valid_types}, got '{fusion_type}'"
            )

        # Build fusion modules based on type
        if fusion_type in ['conv2d', 'shared_conv2d']:
            self._build_conv2d_fusion(
                channels, num_leads, reduction_ratio,
                shared=(fusion_type == 'shared_conv2d')
            )
        elif fusion_type == 'conv3d':
            self._build_conv3d_fusion(channels, num_leads, conv3d_depth)

    def _build_conv2d_fusion(self, channels, num_leads, reduction_ratio, shared):
        """Build conv2d-based fusion modules (existing implementation).

        Args:
            channels: Feature channels
            num_leads: Number of leads
            reduction_ratio: Channel reduction ratio (C → C//reduction_ratio)
            shared: Whether to use shared conv2d across all leads
        """
        self.reduced_channels = channels // reduction_ratio

        if shared:
            # Shared conv2d across all leads
            # More parameter efficient, acts as regularization
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(channels, self.reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.reduced_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Per-lead conv2d
            # Preserves lead-specific information and characteristics
            self.reduce_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, self.reduced_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.reduced_channels),
                    nn.ReLU(inplace=True)
                )
                for _ in range(num_leads)
            ])

        # Mix conv (common for both shared and per-lead)
        # Two 3×3 conv blocks to learn cross-lead patterns
        self.mix_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _build_conv3d_fusion(self, channels, num_leads, depth):
        """Build conv3d-based fusion modules.

        Strategy: 4 leads → 2 leads → 1 lead via Conv3dBlocks with stride

        Args:
            channels: Feature channels
            num_leads: Number of leads (should be 4)
            depth: Number of Conv3dBlocks (should be 2 for 4→2→1 reduction)
        """
        self.conv3d_blocks = nn.ModuleList()

        # Calculate number of reduction stages needed: log2(4) = 2
        num_reduction_stages = int(np.log2(num_leads))

        if depth < num_reduction_stages:
            raise ValueError(
                f"conv3d_depth={depth} is too small for {num_leads} leads. "
                f"Need at least {num_reduction_stages} blocks for {num_leads}→1 reduction."
            )

        for i in range(depth):
            if i < num_reduction_stages:
                # Reduction stage: stride=2 in lead dimension
                stride = (2, 1, 1)
                padding = (0, 1, 1)  # No padding in lead dim
            else:
                # Refinement stage: no reduction
                stride = (1, 1, 1)
                padding = (1, 1, 1)

            self.conv3d_blocks.append(
                Conv3dBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(3, 3, 3),
                    padding=padding,
                    stride=stride,
                    padding_mode='replicate'
                )
            )

    def forward(self, x, batch_size):
        """Forward pass - branches based on fusion_type.

        Args:
            x: (B*4, C, H, W) - Encoder features for all leads concatenated in batch dim
            batch_size: B - Original batch size

        Returns:
            (B*4, C, H, W) - Fused features with same shape as input
        """
        if self.fusion_type in ['conv2d', 'shared_conv2d']:
            return self._forward_conv2d(x, batch_size)
        elif self.fusion_type == 'conv3d':
            return self._forward_conv3d(x, batch_size)

    def _forward_conv2d(self, x, batch_size):
        """Forward pass for conv2d fusion (existing implementation).

        Args:
            x: (B*4, C, H, W) - Encoder features
            batch_size: B - Original batch size

        Returns:
            (B*4, C, H, W) - Fused features with residual
        """
        B = batch_size
        C, H, W = x.shape[1:]

        # Step 1: Separate leads (B*4, C, H, W) → (B, 4, C, H, W)
        x_leads = x.view(B, self.num_leads, C, H, W)

        # Step 2: Reduce each lead's channels
        if self.fusion_type == 'shared_conv2d':
            # Use shared conv2d for all leads
            reduced = [
                self.reduce_conv(x_leads[:, i])  # (B, C//4, H, W) for each lead
                for i in range(self.num_leads)
            ]
        else:  # 'conv2d'
            # Use per-lead conv2d
            reduced = [
                self.reduce_convs[i](x_leads[:, i])  # (B, C//4, H, W) for each lead
                for i in range(self.num_leads)
            ]

        # Step 3: Concatenate reduced features
        concat = torch.cat(reduced, dim=1)  # (B, C, H, W) where C = 4 × C//4

        # Step 4: Mix features to learn cross-lead patterns
        mixed = self.mix_conv(concat)  # (B, C, H, W)

        # Step 5: Broadcast mixed features to all leads
        mixed_broadcast = mixed.unsqueeze(1).expand(B, self.num_leads, C, H, W)

        # Step 6: Residual connection
        fused = x_leads + mixed_broadcast

        # Step 7: Reshape back to (B*4, C, H, W)
        return fused.view(B * self.num_leads, C, H, W)

    def _forward_conv3d(self, x, batch_size):
        """Forward pass for conv3d fusion.

        Args:
            x: (B*4, C, H, W) - Encoder features
            batch_size: B - Original batch size

        Returns:
            (B*4, C, H, W) - Fused features with residual
        """
        B = batch_size
        C, H, W = x.shape[1:]

        # Step 1: Reshape (B*4, C, H, W) → (B, 4, C, H, W)
        x_leads = x.view(B, self.num_leads, C, H, W)

        # Step 2: Transpose for Conv3d (B, 4, C, H, W) → (B, C, 4, H, W)
        # Conv3d expects (B, C, D, H, W) where D is the spatial dimension (lead dimension here)
        x_3d = x_leads.transpose(1, 2)

        # Step 3: Apply Conv3d blocks to reduce lead dimension
        # 4 → 2 → 1 via stride=(2,1,1) in each reduction block
        # Stop when lead dimension becomes 1
        fused = x_3d
        for block in self.conv3d_blocks:
            # Check if we can apply this block (lead dim > 1 or refinement stage)
            lead_dim = fused.shape[2]
            if lead_dim == 1:
                # Lead dimension already reduced to 1, skip further refinement blocks
                break
            fused = block(fused)

        # Step 4: Squeeze lead dimension (B, C, 1, H, W) → (B, C, H, W)
        if fused.shape[2] != 1:
            raise RuntimeError(
                f"Expected lead dimension to be 1 after conv3d blocks, "
                f"got {fused.shape[2]}. Check conv3d_depth configuration."
            )
        fused = fused.squeeze(2)

        # Step 5: Broadcast to all leads (B, C, H, W) → (B, 4, C, H, W)
        fused_broadcast = fused.unsqueeze(1).expand(B, self.num_leads, C, H, W)

        # Step 6: Residual connection
        output = x_leads + fused_broadcast

        # Step 7: Reshape back (B, 4, C, H, W) → (B*4, C, H, W)
        return output.view(B * self.num_leads, C, H, W)


class Net(nn.Module):
    """Stage2 Lead Model with Cross-Lead Feature Fusion.

    This model processes 4 separate lead images through a shared encoder/decoder
    with optional cross-lead feature fusion at each encoder depth level.

    Args:
        encoder_name: Encoder backbone name (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights: Pretrained weights ('imagenet', None)
        fusion_type: Fusion strategy (default: 'conv2d')
            - 'none' or None: No fusion (baseline)
            - 'conv2d': Per-lead reduce conv (recommended for ECG)
            - 'shared_conv2d': Shared reduce conv (parameter efficient)
            - 'conv3d': 3D convolution along lead dimension (NEW)
        fusion_levels: Which encoder levels to fuse [1, 2, 3, 4] (default: [1, 2, 3, 4])
                      1=layer1(64ch), 2=layer2(128ch), 3=layer3(256ch), 4=layer4(512ch)
        num_leads: Number of leads (default: 4)
        loss_weight: Positive class weight for BCE loss (default: 10)
        conv3d_depth: Number of Conv3dBlocks for conv3d fusion (default: 2)

    Input:
        batch['image']: (B, 4, 3, 300, W) - 4 lead images, each 300px height
        batch['pixel']: (B, 4, 1, 300, W) - Ground truth masks (training only)

    Output:
        output['pixel']: (B, 4, 1, 300, W) - Predicted masks (if 'infer' in output_type)
        output['pixel_loss']: scalar - BCE loss (if 'loss' in output_type)
        output['pixel_dice_loss']: scalar - Dice loss (if 'dice_loss' in output_type)

    Examples:
        # Conv2D fusion (recommended)
        model = Net(encoder_name='resnet34', fusion_type='conv2d')

        # Conv3D fusion (NEW)
        model = Net(
            encoder_name='resnet34',
            fusion_type='conv3d',
            fusion_levels=[3, 4],
            conv3d_depth=2
        )

        # No fusion (baseline)
        model = Net(fusion_type='none')

        # Use BCE loss
        model.output_type = ['loss', 'infer']
        output = model(batch)

        # Use Dice loss
        model.output_type = ['dice_loss', 'infer']
        output = model(batch)

        # Use both losses
        model.output_type = ['loss', 'dice_loss', 'infer']
        output = model(batch)
    """

    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        fusion_type='conv2d',
        fusion_levels=[1, 2, 3, 4],
        num_leads=4,
        loss_weight=10,
        conv3d_depth=2,
    ):
        super(Net, self).__init__()

        # Validate and normalize fusion_type
        if fusion_type is None:
            fusion_type = 'none'
        fusion_type = fusion_type.lower()

        valid_types = ['none', 'conv2d', 'shared_conv2d', 'conv3d']
        if fusion_type not in valid_types:
            raise ValueError(
                f"fusion_type must be one of {valid_types}, got '{fusion_type}'"
            )

        # Store configuration
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.fusion_type = fusion_type
        self.fusion_levels = fusion_levels
        self.num_leads = num_leads
        self.loss_weight = loss_weight
        self.conv3d_depth = conv3d_depth

        # Output control
        self.output_type = ['infer', 'loss']

        # Register normalization buffers (ImageNet stats)
        self.register_buffer('D', torch.tensor(0))  # Device marker
        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # Build model components
        self._build_model(encoder_name, encoder_weights)

        # Initialize loss functions
        self.dice_loss_fn = smp.losses.DiceLoss(
            mode='binary',
            from_logits=True,
        )

    def _build_model(self, encoder_name, encoder_weights):
        """Build encoder, decoder, fusion modules, and output head.

        Creates a complete SMP Unet model, then extracts the encoder and decoder
        components. Adds fusion modules for cross-lead feature learning.
        """
        # 1. Create full SMP Unet model
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,  # Not used, we have custom head
            decoder_channels=[256, 128, 64, 32, 16],  # Standard Unet decoder channels
        )

        # 2. Extract encoder and decoder components
        self.encoder = model.encoder
        self.decoder = model.decoder

        # 3. Create cross-lead fusion modules
        self.fusion_modules = nn.ModuleDict()

        if self.fusion_type != 'none':
            # Get encoder output channels: [input, stem, layer1, layer2, layer3, layer4]
            # For ResNet34: [3, 64, 64, 128, 256, 512]
            encoder_channels = self.encoder.out_channels

            for level in self.fusion_levels:
                # level 1 = layer1, level 2 = layer2, etc.
                # Index in encoder.out_channels is level + 1
                channels = encoder_channels[level + 1]

                self.fusion_modules[f'fusion_{level}'] = CrossLeadFusion(
                    channels=channels,
                    num_leads=self.num_leads,
                    fusion_type=self.fusion_type,
                    conv3d_depth=self.conv3d_depth,
                )

        # 4. Create output head (final 1×1 conv)
        # SMP Unet decoder outputs 16 channels (decoder_channels[-1])
        self.pixel_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, batch, L=None):
        """Forward pass.

        Args:
            batch: Dict containing:
                - 'image': (B, 4, 3, H, W) - 4 lead images
                - 'pixel': (B, 4, 1, H, W) - Ground truth masks (training only)
            L: Unused (kept for compatibility)

        Returns:
            output: Dict containing:
                - 'pixel': (B, 4, 1, H, W) - Predicted masks (if 'infer' in output_type)
                - 'pixel_loss': scalar - BCE loss (if 'loss' in output_type)
                - 'pixel_dice_loss': scalar - Dice loss (if 'dice_loss' in output_type)
        """
        device = self.D.device

        # 1. Load and normalize images
        image = batch['image'].to(device)  # (B, 4, 3, H, W)
        B, num_leads, C, H, W = image.shape

        # Normalize with ImageNet stats (broadcast across all leads)
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # 2. Reshape for batch processing: (B, 4, 3, H, W) → (B*4, 3, H, W)
        x = x.view(B * num_leads, C, H, W)

        # 3. Encode through shared encoder
        features = self.encoder(x)
        # features is a list of 6 tensors:
        # [0] = input   (B*4, 3, H, W)
        # [1] = stem    (B*4, 64, H/2, W/2)
        # [2] = layer1  (B*4, 64, H/2, W/2)
        # [3] = layer2  (B*4, 128, H/4, W/4)
        # [4] = layer3  (B*4, 256, H/8, W/8)
        # [5] = layer4  (B*4, 512, H/16, W/16)

        # 4. Apply cross-lead fusion at specified levels
        if self.fusion_type != 'none':
            # Create a mutable copy of features
            fused_features = list(features)

            for level in self.fusion_levels:
                fusion_key = f'fusion_{level}'
                if fusion_key in self.fusion_modules:
                    # Fuse features at this level
                    # features[level + 1] because: level 1 → index 2, level 2 → index 3, etc.
                    fused_features[level + 1] = self.fusion_modules[fusion_key](
                        features[level + 1],
                        batch_size=B
                    )

            features = fused_features

        # 5. Decode through shared decoder
        decoder_output = self.decoder(features)  # (B*4, 16, H, W)

        # 6. Apply output head
        pixel = self.pixel_head(decoder_output)  # (B*4, 1, H, W)

        # 7. Reshape output: (B*4, 1, H, W) → (B, 4, 1, H, W)
        pixel = pixel.view(B, num_leads, 1, H, W)

        # 8. Prepare output dict
        output = {}

        # Prepare flattened tensors for loss calculation (if needed)
        if 'loss' in self.output_type or 'dice_loss' in self.output_type:
            pixel_flat = pixel.view(B * num_leads, 1, H, W)
            target = batch['pixel'].to(device)
            target_flat = target.view(B * num_leads, 1, H, W)

        if 'loss' in self.output_type:
            # Binary cross entropy with logits
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel_flat,
                target_flat,
                pos_weight=torch.tensor([self.loss_weight]).to(device),
            )

        if 'dice_loss' in self.output_type:
            # Dice loss (from segmentation_models_pytorch)
            output['pixel_dice_loss'] = self.dice_loss_fn(pixel_flat, target_flat)

        if 'infer' in self.output_type:
            # Apply sigmoid for inference
            output['pixel'] = torch.sigmoid(pixel)

        return output


def run_check_net():
    """Test function to verify model creation and forward pass.

    Creates models with and without fusion, tests with dummy data,
    and verifies output shapes.
    """
    print("="*60)
    print("Testing Stage2 Lead Model")
    print("="*60)

    H, W = 300, 2176
    batch_size = 2

    # Create dummy batch
    batch = {
        'image': torch.from_numpy(np.random.randint(0, 256, (batch_size, 4, 3, H, W))).byte(),
        'pixel': torch.from_numpy(np.random.rand(batch_size, 4, 1, H, W)).float(),
    }

    # Test configurations
    configs = [
        {
            'name': 'ResNet34 + Fusion (all levels, per-lead conv2d)',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'enable_fusion': True,
            'fusion_levels': [1, 2, 3, 4],
            'shared_reduce_conv': False,
        },
        {
            'name': 'ResNet34 + Fusion (all levels, shared conv2d)',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'enable_fusion': True,
            'fusion_levels': [1, 2, 3, 4],
            'shared_reduce_conv': True,
        },
        {
            'name': 'ResNet34 + No Fusion (baseline)',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'enable_fusion': False,
        },
        {
            'name': 'ResNet34 + Partial Fusion (levels 3, 4, per-lead conv2d)',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'enable_fusion': True,
            'fusion_levels': [3, 4],
            'shared_reduce_conv': False,
        },
    ]

    for config in configs:
        print(f"\n{'-'*60}")
        print(f"Testing: {config['name']}")
        print(f"{'-'*60}")

        # Create model
        net = Net(
            encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'],
            enable_fusion=config['enable_fusion'],
            fusion_levels=config.get('fusion_levels', [1, 2, 3, 4]),
            shared_reduce_conv=config.get('shared_reduce_conv', False),
        ).cuda()

        # Count parameters
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Forward pass with all loss types
        net.output_type = ['loss', 'dice_loss', 'infer']
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = net(batch)

        print("\nOutput shapes:")
        for k, v in output.items():
            if 'loss' not in k:
                print(f"  {k:>15} : {v.shape}")

        print("\nLosses:")
        for k, v in output.items():
            if 'loss' in k:
                print(f"  {k:>15} : {v.item():.6f}")

        # Verify expected shapes
        assert output['pixel'].shape == (batch_size, 4, 1, H, W), \
            f"Expected output shape (2, 4, 1, 300, 2176), got {output['pixel'].shape}"

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == '__main__':
    run_check_net()
