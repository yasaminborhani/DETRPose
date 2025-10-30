# ------------------------------------------------------------------------
# Modified from Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn

class DETRPose(nn.Module):
    def __init__(
        self, 
        backbone, 
        encoder, 
        transformer, 
        is_trainable=True, 
        trainable_energy=False,

        ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.transformer = transformer
        self.layer_loss = torch.zeros(1, dtype=torch.float32)

        def count_params(model):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            return trainable, non_trainable
        
        t_before, nt_before = count_params(self)
        print(f"[Before freeze] Trainable: {t_before:,}, Non-trainable: {nt_before:,}")
        if not is_trainable:
            # freeze backbone, encoder and transformer
            print('Freezing freeze backbone, encoder and transformer...')
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.transformer.parameters():
                p.requires_grad = False

            if trainable_energy:
                # for modules in self.transformer.energy_head, set all params trainable
                energy_head = getattr(self.transformer, "energy_layer", None)
                if energy_head is not None:
                    print('unfreezing energy head...')
                    for p in energy_head.parameters():
                        p.requires_grad = True
                    
                else:
                    # best-effort: try to find any submodule named 'energy' or 'energy_head'
                    for name, module in self.transformer.named_modules():
                        if name.endswith("energy_layer") or name.endswith("energy"):
                            for p in module.parameters():
                                p.requires_grad = True 
            t_after, nt_after = count_params(self)
            print(f"[After freeze]  Trainable: {t_after:,}, Non-trainable: {nt_after:,}")
            print("\n[Trainable parameter names]:")
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print("  ", name)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self

    def forward(self, samples, targets=None):
        feats = self.backbone(samples)
        feats = self.encoder(feats)
        out = self.transformer(feats, targets, samples if self.training else None)
        self.layer_loss = self.transformer.layer_loss
        return out

