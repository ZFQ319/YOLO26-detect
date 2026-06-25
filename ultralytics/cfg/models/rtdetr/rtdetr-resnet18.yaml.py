# Ultralytics 🚀 AGPL-3.0 License
# RT-DETR-ResNet18

nc: 80
scales:
  l: [1.00, 1.00, 1024]
backbone:
  # [from, repeats, module, args]
  # Stem
  - [-1, 1, ResNetLayer18, [3, 64, 1, True]]
  # Layer1
  - [-1, 1, ResNetLayer18, [64, 64, 1, False, 2, 1]]
  # Layer2
  - [-1, 1, ResNetLayer18, [64, 128, 2, False, 2, 1]]
  # Layer3
  - [-1, 1, ResNetLayer18, [128, 256, 2, False, 2, 1]]
  # Layer4
  - [-1, 1, ResNetLayer18, [256, 512, 2, False, 2, 1]]

head:
  # P5 projection
  - [-1, 1, Conv, [256, 1, 1, None, 1, 1, False]]
  # Transformer Encoder
  - [-1, 1, AIFI, [256, 8]]
  # Y5
  - [-1, 1, Conv, [256, 1, 1]]
  # ---------------- FPN ----------------
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # Layer3 output (256 channels)
  - [3, 1, Conv, [256, 1, 1, None, 1, 1, False]]
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]
  - [-1, 1, Conv, [256, 1, 1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # Layer2 output (128 channels)
  - [2, 1, Conv, [256, 1, 1, None, 1, 1, False]]
  - [[-2, -1], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]
  # ---------------- PAN ----------------
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 7], 1, Concat, [1]]
  - [-1, 3, RepC3, [256]]

  # Decoder
  - [[16, 19, 22], 1, RTDETRDecoder, [nc]]