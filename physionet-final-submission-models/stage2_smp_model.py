"""
Stage2 Model with segmentation_models_pytorch (SMP) support

This module provides a flexible Net class that supports:
- Multiple encoders from timm/smp (resnet34, efficientnet, etc.)
- Multiple decoder architectures (Unet, FPN, PAN)
- Optional CoordConv integration for position-aware decoding
- Compatible interface with original stage2_model.py

Usage:
    # Standard SMP model
    model = Net(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        decoder_name='unet',
        use_coord_conv=False
    )

    # CoordConv model (uses custom decoder from original implementation)
    model = Net(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        decoder_name='unet',  # ignored when use_coord_conv=True
        use_coord_conv=True
    )
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

# Import CoordConv decoder from original implementation
# This is used when use_coord_conv=True


class MyCoordDecoderBlock(nn.Module):
    """CoordConv decoder block from original stage2_model.py

    Adds coordinate information (x, y) to each decoder block for position-aware decoding.
    """
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel,
            scale=2,
    ):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel+2, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        # Add coordinate information
        b, c, h, w = x.shape
        coordx, coordy = torch.meshgrid(
            torch.linspace(-2, 2, w, dtype=x.dtype, device=x.device),
            torch.linspace(-2, 2, h, dtype=x.dtype, device=x.device),
            indexing='xy'
        )
        coordxy = torch.stack([coordx, coordy], dim=1).reshape(1,2,h,w).repeat(b,1,1,1)
        x = torch.cat([x, coordxy], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyCoordUnetDecoder(nn.Module):
    """CoordConv U-Net decoder from original stage2_model.py

    Args:
        in_channel: Input channel dimension from encoder bottleneck
        skip_channel: List of skip connection channels (should match depth)
        out_channel: List of output channels for each decoder block (should match depth)
        scale: List of upsampling scales for each block (should match depth)
        depth: Number of decoder blocks (4 or 5). If provided, overrides out_channel.
               - depth=4: [256, 128, 64, 32] (original, uses layer1-4)
               - depth=5: [256, 128, 64, 32, 16] (SMP-compatible, uses stem+layer1-4)
    """
    def __init__(
            self,
            in_channel,
            skip_channel,
            out_channel=None,
            scale=None,
            depth=4
    ):
        super().__init__()
        self.center = nn.Identity()
        self.depth = depth

        # Auto-configure channels based on depth
        if out_channel is None:
            if depth == 4:
                out_channel = [256, 128, 64, 32]
            elif depth == 5:
                out_channel = [256, 128, 64, 32, 16]
            else:
                raise ValueError(f"depth must be 4 or 5, got {depth}")

        # Auto-configure scale based on depth
        if scale is None:
            scale = [2] * depth

        # Validate
        if len(out_channel) != depth:
            raise ValueError(f"out_channel length ({len(out_channel)}) must match depth ({depth})")
        if len(scale) != depth:
            raise ValueError(f"scale length ({len(scale)}) must match depth ({depth})")
        if len(skip_channel) != depth:
            raise ValueError(f"skip_channel length ({len(skip_channel)}) must match depth ({depth})")

        i_channel = [in_channel, ] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyCoordDecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


class Net(nn.Module):
    """Flexible Stage2 segmentation model with SMP support

    Args:
        encoder_name: Encoder backbone name (e.g., 'resnet34', 'efficientnet-b0', 'timm-resnest50d')
        encoder_weights: Pretrained weights ('imagenet', 'imagenet21k', 'ssl', 'swsl', None)
        decoder_name: Decoder architecture ('unet', 'fpn', 'pan')
        use_coord_conv: If True, use CoordConv decoder (ignores decoder_name)
        coord_decoder_depth: Depth for CoordConv decoder (4 or 5). Only used when use_coord_conv=True.
                            - 4: Original 4-block decoder using layer1-4 (default)
                            - 5: 5-block decoder using stem+layer1-4 (SMP-compatible)
        pretrained: Legacy parameter for backward compatibility (maps to encoder_weights='imagenet')

    Example:
        # Standard Unet with ResNet34 encoder
        model = Net(encoder_name='resnet34', encoder_weights='imagenet', decoder_name='unet')

        # FPN with EfficientNet encoder
        model = Net(encoder_name='efficientnet-b3', encoder_weights='imagenet', decoder_name='fpn')

        # CoordConv Unet with 4 blocks (original)
        model = Net(encoder_name='resnet34', encoder_weights='imagenet', use_coord_conv=True, coord_decoder_depth=4)

        # CoordConv Unet with 5 blocks (SMP-compatible)
        model = Net(encoder_name='resnet34', encoder_weights='imagenet', use_coord_conv=True, coord_decoder_depth=5)
    """

    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        decoder_name='unet',
        use_coord_conv=False,
        coord_decoder_depth=4,
        loss_weight=10,
        pretrained=True  # For backward compatibility
    ):
        super(Net, self).__init__()

        # Handle backward compatibility
        if encoder_weights is None and pretrained:
            encoder_weights = 'imagenet'

        # Validate decoder_name
        if decoder_name not in ['unet', 'fpn', 'pan']:
            raise ValueError(f"decoder_name must be 'unet', 'fpn', or 'pan', got '{decoder_name}'")

        # Validate coord_decoder_depth
        if coord_decoder_depth not in [4, 5]:
            raise ValueError(f"coord_decoder_depth must be 4 or 5, got {coord_decoder_depth}")

        # Store config
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.decoder_name = decoder_name
        self.use_coord_conv = use_coord_conv
        self.coord_decoder_depth = coord_decoder_depth
        self.loss_weight = loss_weight

        # Fixed decoder channels
        # Note: For SMP decoders, we need 5 channels (depth=5)
        # CoordConv can now use 4 or 5 blocks
        self.decoder_channels_coord_4 = [256, 128, 64, 32]  # For CoordConv decoder (4 blocks)
        self.decoder_channels_coord_5 = [256, 128, 64, 32, 16]  # For CoordConv decoder (5 blocks)
        self.decoder_channels_smp = [256, 128, 64, 32, 16]  # For SMP decoders (5 blocks)

        # Register buffers for normalization
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # Build model
        if use_coord_conv:
            # Use custom CoordConv decoder
            self._build_coord_conv_model(encoder_name, encoder_weights)

            # Create auxiliary pixel heads (one per intermediate block)
            self.aux_pixel_heads = nn.ModuleList([
                nn.Conv2d(ch + 1, 4, kernel_size=1)  # ch + 1 for coordy
                for ch in self.decoder_channels[:-1]  # Exclude final block
            ])
        else:
            # Use SMP decoder
            self._build_smp_model(encoder_name, encoder_weights, decoder_name)
            # SMP decoders don't support aux features
            self.aux_pixel_heads = None

        # Segmentation head (final conv layer with coordy)
        # Input: decoder output (final_decoder_channels) + coordy (1 channel) = final_decoder_channels + 1
        # Output: 4 channels (4 series)
        self.pixel = nn.Conv2d(self.final_decoder_channels + 1, 4, kernel_size=1)

    def _build_coord_conv_model(self, encoder_name, encoder_weights):
        """Build model with CoordConv decoder"""
        # Get encoder from SMP
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )

        # Get encoder output channels
        # SMP encoder.out_channels = [3, 64, 64, 128, 256, 512] for ResNet18/34
        #   [0] = input channels (3)
        #   [1] = stem output (conv1+bn1+relu, before maxpool) (64)
        #   [2] = layer1 output (64)
        #   [3] = layer2 output (128)
        #   [4] = layer3 output (256)
        #   [5] = layer4 output (512)
        encoder_channels = self.encoder.out_channels

        if self.coord_decoder_depth == 4:
            # Original 4-block decoder: uses layer1-4 outputs only (skip stem)
            # encoder_dim = [64, 128, 256, 512] from layer1-4
            encoder_dim = encoder_channels[2:]  # Skip input and stem
            decoder_channels = self.decoder_channels_coord_4
            skip_channel = encoder_dim[:-1][::-1] + [0]  # [256, 128, 64, 0]
        elif self.coord_decoder_depth == 5:
            # 5-block decoder: uses stem + layer1-4 outputs (SMP-compatible)
            # encoder_dim = [64, 64, 128, 256, 512] from stem, layer1-4
            encoder_dim = encoder_channels[1:]  # Skip only input
            decoder_channels = self.decoder_channels_coord_5
            skip_channel = encoder_dim[:-1][::-1] + [0]  # [256, 128, 64, 64, 0]
        else:
            raise ValueError(f"coord_decoder_depth must be 4 or 5, got {self.coord_decoder_depth}")

        # Store decoder_channels for creating auxiliary heads
        self.decoder_channels = decoder_channels

        self.decoder = MyCoordUnetDecoder(
            in_channel=encoder_dim[-1],  # 512 (layer4 output)
            skip_channel=skip_channel,
            out_channel=None,  # Auto-configured by depth parameter
            scale=None,        # Auto-configured by depth parameter
            depth=self.coord_decoder_depth
        )

        self.use_smp_decoder = False
        self.final_decoder_channels = decoder_channels[-1]  # 32 (depth=4) or 16 (depth=5)

    def _build_smp_model(self, encoder_name, encoder_weights, decoder_name):
        """Build model with SMP decoder"""
        # Create SMP model
        if decoder_name == 'unet':
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=4,  # Not used (we use custom segmentation head)
                decoder_channels=self.decoder_channels_smp,  # [256, 128, 64, 32, 16]
            )
            self.final_decoder_channels = self.decoder_channels_smp[-1]  # 16
        elif decoder_name == 'fpn':
            model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=4,  # Not used
                decoder_pyramid_channels=256,
                decoder_segmentation_channels=128,
            )
            # FPN output channels = decoder_segmentation_channels
            self.final_decoder_channels = 128
        elif decoder_name == 'pan':
            model = smp.PAN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=4,  # Not used
                decoder_channels=32,  # PAN uses single decoder_channels value
            )
            # PAN output channels = decoder_channels
            self.final_decoder_channels = 32

        # Extract encoder and decoder from SMP model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.use_smp_decoder = True

    def forward(self, batch, L=None):
        """Forward pass

        Args:
            batch: Dict containing 'image' (B, 3, H, W) and optionally 'pixel' (B, 4, H, W) for training
            L: Target length for series (not used in this version)

        Returns:
            output: Dict containing:
                - 'pixel': Predicted pixel probabilities (B, 4, H, W) if 'infer' in output_type
                - 'pixel_loss': Binary cross entropy loss if 'loss' in output_type
        """
        device = self.D.device

        # Load and normalize image
        image = batch['image'].to(device)
        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        # Encoder
        features = self.encoder(x)
        # features from SMP encoder: [input, stem, layer1, layer2, layer3, layer4]
        #                           = [3, 64, 64, 128, 256, 512] for ResNet18/34

        # Decoder
        if self.use_smp_decoder:
            # SMP decoder expects all features including stem
            decoder_output = self.decoder(features)
            intermediate_outputs = None  # SMP decoders don't expose intermediate outputs
        else:
            # CoordConv decoder
            if self.coord_decoder_depth == 4:
                # 4-block decoder: uses layer1-4 features (skip input and stem)
                # Match original stage2_model.py behavior
                features_for_decoder = features[2:]  # [layer1, layer2, layer3, layer4]
                skip_features = features_for_decoder[:-1][::-1] + [None]  # [layer3, layer2, layer1, None]
            elif self.coord_decoder_depth == 5:
                # 5-block decoder: uses stem + layer1-4 features (skip only input)
                features_for_decoder = features[1:]  # [stem, layer1, layer2, layer3, layer4]
                skip_features = features_for_decoder[:-1][::-1] + [None]  # [layer3, layer2, layer1, stem, None]
            else:
                raise ValueError(f"coord_decoder_depth must be 4 or 5, got {self.coord_decoder_depth}")

            last, decode = self.decoder(
                feature=features_for_decoder[-1],  # layer4 (always the last feature)
                skip=skip_features
            )
            decoder_output = last
            intermediate_outputs = decode  # Save for aux features

        # Error handling for incompatible modes
        if self.use_smp_decoder and ('aux_loss' in self.output_type or 'infer_all' in self.output_type):
            raise ValueError(
                "aux_loss and infer_all are only supported with CoordConv decoder. "
                "Set use_coord_conv=True or remove these from output_type."
            )

        # Create coordy for final layer (y-coordinate normalized to [-1, 1])
        # Match the decoder output spatial dimensions
        _, _, H_out, W_out = decoder_output.shape
        coordy = torch.arange(H_out, device=device).reshape(1, 1, H_out, 1).repeat(B, 1, 1, W_out)
        coordy = coordy / (H_out - 1) * 2 - 1

        # Segmentation head with coordy
        # Concatenate decoder output with y-coordinate
        last = torch.cat([decoder_output, coordy], dim=1)
        pixel = self.pixel(last)

        # Upsample pixel to match input image size if needed
        # (FPN and PAN output at lower resolution than input)
        if pixel.shape[2:] != (H, W):
            pixel = F.interpolate(pixel, size=(H, W), mode='bilinear', align_corners=False)

        # Prepare output
        output = {}

        if 'loss' in self.output_type:
            # Binary cross entropy loss with positive class weighting
            output['pixel_loss'] = F.binary_cross_entropy_with_logits(
                pixel,
                batch['pixel'].to(device),
                pos_weight=torch.tensor([self.loss_weight]).to(device),
            )

        if 'aux_loss' in self.output_type:
            if intermediate_outputs is None:
                raise ValueError("aux_loss requires CoordConv decoder")

            # Calculate weights
            num_blocks = len(intermediate_outputs)
            aux_weight = 0.5 / (num_blocks - 1)
            final_weight = 0.5

            target = batch['pixel'].to(device)
            target_size = target.shape[2:]

            total_aux_loss = 0.0

            # Process intermediate blocks (exclude final)
            for i, decoder_out in enumerate(intermediate_outputs[:-1]):
                # Create coordy for this resolution
                _, _, H_i, W_i = decoder_out.shape
                coordy_i = torch.arange(H_i, device=device).reshape(1, 1, H_i, 1).repeat(B, 1, 1, W_i)
                coordy_i = coordy_i / (H_i - 1) * 2 - 1

                # Concat and apply aux head
                decoder_with_coord = torch.cat([decoder_out, coordy_i], dim=1)
                pixel_aux = self.aux_pixel_heads[i](decoder_with_coord)

                # Upsample to target size
                pixel_aux = F.interpolate(pixel_aux, size=target_size, mode='bilinear', align_corners=False)

                # Calculate BCE loss
                loss_aux = F.binary_cross_entropy_with_logits(
                    pixel_aux, target,
                    pos_weight=torch.tensor([self.loss_weight]).to(device),
                )

                total_aux_loss += aux_weight * loss_aux

            # Add final block loss (weighted)
            final_loss = F.binary_cross_entropy_with_logits(
                pixel, target,
                pos_weight=torch.tensor([self.loss_weight]).to(device),
            )
            total_aux_loss += final_weight * final_loss

            # Override pixel_loss with weighted sum
            output['pixel_loss'] = total_aux_loss

        if 'infer' in self.output_type:
            # Sigmoid activation for inference
            output['pixel'] = torch.sigmoid(pixel)

        if 'infer_all' in self.output_type:
            if intermediate_outputs is None:
                raise ValueError("infer_all requires CoordConv decoder")

            target_size = (H, W)

            # Process ALL blocks (including final)
            for i, decoder_out in enumerate(intermediate_outputs):
                # Create coordy
                _, _, H_i, W_i = decoder_out.shape
                coordy_i = torch.arange(H_i, device=device).reshape(1, 1, H_i, 1).repeat(B, 1, 1, W_i)
                coordy_i = coordy_i / (H_i - 1) * 2 - 1

                decoder_with_coord = torch.cat([decoder_out, coordy_i], dim=1)

                # Apply appropriate head
                if i < len(intermediate_outputs) - 1:
                    pixel_i = self.aux_pixel_heads[i](decoder_with_coord)
                else:
                    pixel_i = self.pixel(decoder_with_coord)

                # Upsample
                pixel_i = F.interpolate(pixel_i, size=target_size, mode='bilinear', align_corners=False)

                # Store with sigmoid
                output[f'pixel_{i}'] = torch.sigmoid(pixel_i)

            # Alias final output as 'pixel'
            output['pixel'] = output[f'pixel_{len(intermediate_outputs) - 1}']

        return output


def run_check_net():
    """Test function to verify model creation and forward pass"""
    print("="*60)
    print("Testing Stage2 SMP Model")
    print("="*60)

    H, W = 320, 320
    batch_size = 2

    # Dummy data
    batch = {
        'image': torch.from_numpy(np.random.randint(0, 256, (batch_size, 3, H, W))).byte(),
        'pixel': torch.from_numpy(np.random.rand(batch_size, 4, H, W)).float(),
    }

    # Test configurations
    configs = [
        {
            'name': 'Unet + ResNet34',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'decoder_name': 'unet',
            'use_coord_conv': False,
        },
        {
            'name': 'FPN + ResNet34',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'decoder_name': 'fpn',
            'use_coord_conv': False,
        },
        {
            'name': 'PAN + ResNet34',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'decoder_name': 'pan',
            'use_coord_conv': False,
        },
        {
            'name': 'CoordConv Unet (4 blocks) + ResNet34',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'decoder_name': 'unet',  # Ignored
            'use_coord_conv': True,
            'coord_decoder_depth': 4,
        },
        {
            'name': 'CoordConv Unet (5 blocks) + ResNet34',
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'decoder_name': 'unet',  # Ignored
            'use_coord_conv': True,
            'coord_decoder_depth': 5,
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
            decoder_name=config['decoder_name'],
            use_coord_conv=config['use_coord_conv'],
            coord_decoder_depth=config.get('coord_decoder_depth', 4),
        ).cuda()

        # Count parameters
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Forward pass
        net.output_type = ['loss', 'infer']
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

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)


if __name__ == '__main__':
    run_check_net()
