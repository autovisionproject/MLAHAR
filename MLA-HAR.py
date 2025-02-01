import math
import torch
import torch.nn as nn
import numpy as np

#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
#np.random.seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.eps = eps

    def forward(self, queries, keys, values):
        # Input: (batch_size, num_heads, sequence_length, head_dim)
        Q = queries  # (B, H, L, D)
        K = keys     # (B, H, L, D)
        V = values   # (B, H, L, D)
        KV = torch.einsum("bhld,bhmd->bhdm", K, V)
        attention_scores = torch.einsum("bhld,bhld->bhl", Q, K)
        Z = 1 / (attention_scores.sum(dim=-1, keepdim=True) + self.eps)
        V_out = torch.einsum("bhl,bhdm,bhl->bhld", attention_scores, KV, Z)
        return V_out, attention_scores

class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear transformations
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        # Attention
        self.attention = LinearAttention()


    def forward(self, queries, keys, values):
        B, L, D = queries.shape
        Q = self.linear_q(queries).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        K = self.linear_k(keys).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   
        V = self.linear_v(values).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out, attention_weights = self.attention(Q, K, V)  
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.d_model)
        #out= self.linear_out(out)
        return out, attention_weights



class GaussianNoise(nn.Module):
    def __init__(self, std_max=0.2, std_min=0.01, total_epochs=10):
        """
        Gaussian noise layer with cosine noise scheduling.
        Args:
            std_max: Maximum standard deviation for noise.
            std_min: Minimum standard deviation for noise.
            total_epochs: Total number of epochs to adjust the noise.
        """
        super(GaussianNoise, self).__init__()
        self.std_max = std_max
        self.std_min = std_min
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def update_epoch(self, epoch):
        """Update the current epoch for noise scheduling."""
        self.current_epoch = epoch

    def get_std(self):
        """Calculate the standard deviation based on the cosine schedule."""
        cos_factor = 0.5 * (1 + math.cos(math.pi * self.current_epoch / self.total_epochs))
        return self.std_min + (self.std_max - self.std_min) * cos_factor

    def forward(self, x,training=False):
        if training:  # Add noise only during training
            std = self.get_std()
            noise = torch.randn_like(x) * std
            #print(noise)
            return x + noise
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len]
        pe = pe.squeeze(1)
        pe = pe.unsqueeze(0)
        pe = pe.expand(x.size(0), -1, -1)
        x = x + pe
        return x


class MLA(nn.Module):
    def __init__(self, output, parms_diam, dropout, num_heads, batch_size, diam_internal, weight_decay, total_epochs,window_size):
        super(MLA, self).__init__()

        self.conv1 = nn.Conv1d(parms_diam, diam_internal[0], kernel_size=3, stride=1, padding=1)

        self.PositionalEncoding = PositionalEncoding(diam_internal[0] ,batch_size)
        self.PositionalEncoding.requires_grad = False

        self.gaussian_noise = GaussianNoise(std_max=0.1, std_min=0.01, total_epochs=total_epochs)
        self.multi_attention1 = MultiHeadLinearAttention(diam_internal[0], num_heads)
        self.multi_attention2 = MultiHeadLinearAttention(diam_internal[0], num_heads)

        self.norm1 = nn.LayerNorm(diam_internal[0])
        self.norm2 = nn.LayerNorm(diam_internal[0])
        self.norm3 = nn.LayerNorm(diam_internal[0])

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(parms_diam, diam_internal[0])
        self.fc2 = nn.Linear(diam_internal[0],diam_internal[0])
        self.fc3 = nn.Linear(diam_internal[0],diam_internal[0])
        #self.fc4 = nn.Linear(diam_internal[0]*window_size, diam_internal[1])
        self.fc4 = nn.Linear(diam_internal[0], diam_internal[1])
        #self.fc5 = nn.Linear(diam_internal[1], diam_internal[2])
        self.fc5 = nn.Linear(diam_internal[1], output)
        self.activation = nn.GELU()

        self.weight_decay = weight_decay

    def update_epoch(self, epoch):
        """Update the epoch for the GaussianNoise layer."""
        self.gaussian_noise.update_epoch(epoch)

    def forward(self, x,training=False):
        #print(x.shape)
        x1 = self.activation(self.conv1(x.permute(0,2,1))).permute(0,2,1)
        x2 = self.activation(self.fc1(x))
        #x  = self.PositionalEncoding(x1+x2)
        #x  = self.norm1(x + self.gaussian_noise(x,training=training))
        x  = self.norm1(self.gaussian_noise(x1+x2,training=training))
        x  = self.PositionalEncoding(x)
        attention1,_ = self.multi_attention1(x, x, x)
        f1 = self.activation(self.fc2(x))
        #print("attention1",f1.shape,attention1.shape)
        f2=self.dropout(self.norm2(f1 + attention1))

        attention2,_ = self.multi_attention2(f2, f2, f2)
        f1 = self.activation(self.fc3(f1))
        f3=self.dropout(self.norm3(f1 + attention2))
        #f2 = torch.flatten(f2, start_dim=1)
        f3 = self.activation(self.fc4(f3))
        #f2 = self.activation(self.fc5(f2))
        f3 = self.fc5(f3)
        f3=f3[:,-1,:]
        return f3

