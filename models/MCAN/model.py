# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAugmentationModule(nn.Module):
    """논문의 3.2 섹션, Fig 4에 해당하는 특징 증강 모듈."""
    def __init__(self):
        super(FeatureAugmentationModule, self).__init__()
        self.conv_f1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn_f1 = nn.BatchNorm2d(16)
        self.pool_f1 = nn.MaxPool2d(2)
        self.conv_h1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn_h1 = nn.BatchNorm2d(16)
        self.pool_h1 = nn.MaxPool2d(2)
        self.conv_f2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn_f2 = nn.BatchNorm2d(32)
        self.pool_f2 = nn.MaxPool2d(2)
        self.conv_fh = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.bn_fh = nn.BatchNorm2d(64)
        self.pool_fh = nn.MaxPool2d(2)

    def forward(self, x_f, x_h):
        x_f_prime = self.pool_f1(F.relu(self.bn_f1(self.conv_f1(x_f))))
        x_f_double_prime = self.pool_f2(F.relu(self.bn_f2(self.conv_f2(x_f_prime))))
        x_h_prime = self.pool_h1(F.relu(self.bn_h1(self.conv_h1(x_h))))
        x_fh = torch.cat([x_f_double_prime, x_h_prime], dim=1) # 초기 융합
        x_fh_prime = self.pool_fh(F.relu(self.bn_fh(self.conv_fh(x_fh))))
        return x_fh_prime

class WeightedAttentionModule(nn.Module):
    """논문의 3.3 섹션, Fig 5에 해당하는 가중치 어텐션 모듈."""
    def __init__(self, input_dim):
        super(WeightedAttentionModule, self).__init__()
        self.input_dim = input_dim
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.input_dim ** 0.5)
        attn_dist = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_dist, v)
        weighted_sum = (1 - self.beta) * x + self.beta * attn_output # 가중 합산
        return weighted_sum

class MCAN(nn.Module):
    """논문에서 제안된 MCAN 모델의 전체 아키텍처."""
    def __init__(self, num_styles, style_embedding_dim=16):
        super(MCAN, self).__init__()
        self.feature_augmentation = FeatureAugmentationModule()
        self.attention_input_dim = 64 * 7 # (C * H) after reshaping
        self.weighted_attention = WeightedAttentionModule(input_dim=self.attention_input_dim)
        self.blstm = nn.LSTM(
            input_size=self.attention_input_dim,
            hidden_size=128, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )
        self.style_embedding = nn.Embedding(num_styles, style_embedding_dim) # 스타일 임베딩 모듈
        regressor_input_dim = 256 + style_embedding_dim
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, 128),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, filter_bank, handcrafted_features, style_indices):
        augmented_features = self.feature_augmentation(filter_bank, handcrafted_features)
        batch_size = augmented_features.size(0)
        # (B, C, H, W) -> (B, W, C*H)로 변경하여 시퀀스 데이터로 변환
        x = augmented_features.permute(0, 3, 1, 2).reshape(batch_size, 30, -1)
        x = self.weighted_attention(x)
        _, (h_n, _) = self.blstm(x)
        blstm_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        style_vec = self.style_embedding(style_indices)
        final_feature_vector = torch.cat([blstm_output, style_vec], dim=1)
        predictions = self.regressor(final_feature_vector)
        return predictions