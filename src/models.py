import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = efficientnet_b0(weights=weights)
        
        for param in backbone.parameters():
            param.requires_grad = False
        
        for param in backbone.features[6:].parameters():
            param.requires_grad = True
        
        # Сохраняем части backbone
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.features[0](x)
        x = self.features[1](x)
        feat1 = self.features[2](x)
        feat2 = self.features[3](feat1)
        feat3 = self.features[4](feat2)
        x = self.features[5](feat3)
        x = self.features[6](x)
        x = self.features[7](x)
        feat4 = self.features[8](x)
        
        return [feat1, feat2, feat3, feat4]

class SliceEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.backbone = FeatureExtractor()
        
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(24, 64, 1), 
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.Conv2d(40, 64, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.Conv2d(80, 64, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            nn.Sequential(
                nn.Conv2d(1280, 64, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        ])
        
        self.agg = nn.Sequential(
            nn.Linear(64 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x):
        
        features = self.backbone(x)
        
        head_outputs = []
        for feat, head in zip(features, self.heads):
            out = head(feat)
            head_outputs.append(out)
        
        combined = torch.cat(head_outputs, dim=1)
        return self.agg(combined)

class CTClassifier(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = SliceEncoder(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb).squeeze(-1)
        return logits
        
class SliceFormer3D(nn.Module):
    def __init__(
        self,
        slice_encoder,
        embed_dim=128,
        num_heads=8,
        num_transformer_layers=2,
        max_slices=800,
        dropout=0.1
    ):
        super().__init__()
        self.slice_encoder = slice_encoder

        
        self.local_context = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.ReLU()
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.pos_embed = nn.Parameter(torch.randn(1, max_slices + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        
        x_flat = x.view(B * N, C, H, W)
        slice_embeds = self.slice_encoder(x_flat)
        slice_embeds = slice_embeds.view(B, N, -1)
        
        slice_embeds = slice_embeds.permute(0, 2, 1)
        slice_embeds = self.local_context(slice_embeds)
        slice_embeds = slice_embeds.permute(0, 2, 1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, slice_embeds], dim=1)

        if N + 1 > self.pos_embed.shape[1]:
            raise ValueError(f"Слишком много срезов! Максимум: {self.pos_embed.shape[1] - 1}")
        x = x + self.pos_embed[:, :N+1, :]
        
        x = self.transformer(x)
        
        cls_output = x[:, 0]
        logits = self.classifier(cls_output).squeeze(-1)
        
        return logits