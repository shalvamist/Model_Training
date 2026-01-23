import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# V17 Advanced Components
from .advanced_components import RevIN, KANLinear, SwiGLU

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dynamic, x_static):
        context = x_static.unsqueeze(1) 
        attn_out, _ = self.mha(query=x_dynamic, key=context, value=context)
        return self.norm(x_dynamic + self.dropout(attn_out))

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, n_experts=4, hidden_dim=64, use_kan=False):
        super().__init__()
        self.use_kan = use_kan
        
        linear_layer = KANLinear if use_kan else nn.Linear
        
        # Build Experts
        # If KAN, we might not need GELU in between if grid size is sufficient, 
        # but maintaining structure for consistency.
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            if use_kan:
                # KAN implies learnable activation, so pure KAN-KAN might be enough
                self.experts.append(nn.Sequential(
                    KANLinear(input_dim, hidden_dim),
                    KANLinear(hidden_dim, output_dim)
                ))
            else:
                self.experts.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim)
                ))
            
        self.gate = nn.Linear(input_dim, n_experts)
        
    def forward(self, x):
        gate_logits = self.gate(x) 
        weights = F.softmax(gate_logits, dim=-1)
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1) 
        weighted_out = torch.sum(expert_outputs * weights.unsqueeze(-1), dim=1) 
        return weighted_out

class HybridEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers=1, trans_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        trans_in = self.pos_encoder(lstm_out)
        trans_out = self.transformer(trans_in)
        return self.norm(trans_out)

class ResidualBlock(nn.Module):
    """
    Simple Residual MLP Block: x + MLP(x)
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.norm(x + self.net(x))

class HybridJointNetwork(nn.Module):
    """
    Universal Architecture (formerly V11/V12/V13).
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        dropout = config.get('dropout', 0.2)
        
        self.dow_embed = nn.Embedding(7, 4)
        self.moy_embed = nn.Embedding(12, 4)
        
        if 'input_dim' in config:
            dyn_dim = config['input_dim']
        else:
            dyn_dim = len(config['feature_cols'])
            
        if 'static_dim' in config:
            static_dim = config['static_dim'] + 8 
        else:
            static_dim = len(config.get('static_cols', [])) + 8
        
        # Encoder
        self.hybrid_encoder = HybridEncoder(
            dyn_dim, 
            self.hidden_size, 
            lstm_layers=config.get('lstm_layers', 2),
            trans_layers=config.get('trans_layers', 2),
            dropout=dropout
        )
        
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.cross_att = CrossAttention(self.hidden_size, num_heads=4, dropout=dropout)
        self.pool_query = nn.Linear(self.hidden_size, 1)
        self.res_block = ResidualBlock(self.hidden_size, dropout=dropout)
        
        n_experts = config.get('n_experts', 4)
        
        # Heads
        self.head_1_reg = MixtureOfExperts(self.hidden_size, output_dim=1, n_experts=n_experts, hidden_dim=self.hidden_size // 2)
        self.head_1_cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 3) 
        )
        
        self.head_2_reg = MixtureOfExperts(self.hidden_size, output_dim=1, n_experts=n_experts, hidden_dim=self.hidden_size // 2)
        self.head_2_cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 3) 
        )

    def forward(self, x_dyn, x_stat, x_time):
        dow = self.dow_embed(x_time[:, 0])
        moy = self.moy_embed(x_time[:, 1])
        static_combined = torch.cat([x_stat, dow, moy], dim=1)
        static_emb = self.static_proj(static_combined) 
        
        seq_rep = self.hybrid_encoder(x_dyn)
        fused_seq = self.cross_att(seq_rep, static_emb)
        
        weights = F.softmax(self.pool_query(fused_seq), dim=1)
        context_vector = torch.sum(fused_seq * weights, dim=1)
        deep_context = self.res_block(context_vector)
        
        reg_1 = self.head_1_reg(deep_context)
        cls_1 = self.head_1_cls(deep_context)
        reg_2 = self.head_2_reg(deep_context)
        cls_2 = self.head_2_cls(deep_context)
        
        return reg_1, cls_1, reg_2, cls_2


class ExperimentalNetwork(nn.Module):
    """
    V17 "Cutting Edge" Architecture.
    Features:
    - RevIN (Distribution Shift Robustness)
    - Bidirectional LSTM (Full Context)
    - SwiGLU Gating
    - KAN Expert Heads (Spline-based Activation)
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        dropout = config.get('dropout', 0.2)
        
        # 1. Input Dimensions
        if 'input_dim' in config:
            dyn_dim = config['input_dim']
        else:
            dyn_dim = len(config['feature_cols'])
            
        if 'static_dim' in config:
            static_dim = config['static_dim'] + 8 
        else:
            static_dim = len(config.get('static_cols', [])) + 8

        # 2. RevIN (Optional by config, default True for V17)
        self.use_revin = config.get('use_revin', True)
        if self.use_revin:
            self.revin = RevIN(dyn_dim)
            
        # 3. Dynamic Encoder (Bi-LSTM + Transformer)
        self.input_proj = nn.Linear(dyn_dim, self.hidden_size)
        
        # Bi-Directional LSTM
        # If bidirectional, output dim is hidden_size * 2
        self.bidirectional = config.get('bidirectional', True)
        self.lstm = nn.LSTM(
            self.hidden_size, 
            self.hidden_size, 
            num_layers=config.get('lstm_layers', 2), 
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        lstm_out_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        
        # Projection back to hidden size for Transformer
        self.lstm_proj = nn.Linear(lstm_out_dim, self.hidden_size)
        
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        
        # Transformer
        # SwiGLU in FFN? Pytorch 2.0+ supports activation="swiglu", else "gelu"
        # We stick to standard TransformerEncoder for stability, but could insert custom layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=4, 
            dim_feedforward=self.hidden_size * 4, 
            dropout=dropout, 
            batch_first=True, 
            norm_first=True, 
            activation="gelu" 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.get('trans_layers', 2))
        self.norm = nn.LayerNorm(self.hidden_size)
        
        # 4. Static Encoder (SwiGLU Enhanced)
        self.dow_embed = nn.Embedding(7, 4)
        self.moy_embed = nn.Embedding(12, 4)
        
        self.static_net = nn.Sequential(
            SwiGLU(static_dim, self.hidden_size, self.hidden_size),
            nn.Dropout(dropout)
        )
        
        # 5. Fusion
        self.cross_att = CrossAttention(self.hidden_size, num_heads=4, dropout=dropout)
        self.pool_query = nn.Linear(self.hidden_size, 1)
        
        # 6. Heads (KAN Enhanced)
        use_kan_experts = config.get('use_kan', True)
        n_experts = config.get('n_experts', 4)
        
        self.head_1_reg = MixtureOfExperts(self.hidden_size, 1, n_experts, self.hidden_size // 2, use_kan=use_kan_experts)
        self.head_1_cls = MixtureOfExperts(self.hidden_size, 3, n_experts, self.hidden_size // 2, use_kan=use_kan_experts)
        
        self.head_2_reg = MixtureOfExperts(self.hidden_size, 1, n_experts, self.hidden_size // 2, use_kan=use_kan_experts)
        self.head_2_cls = MixtureOfExperts(self.hidden_size, 3, n_experts, self.hidden_size // 2, use_kan=use_kan_experts)

    def forward(self, x_dyn, x_stat, x_time):
        # 1. RevIN Normalization
        if self.use_revin:
            x_dyn = self.revin(x_dyn, 'norm')
            
        # 2. Dynamic Encoding
        x = self.input_proj(x_dyn)
        x_lstm, _ = self.lstm(x)
        x_lstm = self.lstm_proj(x_lstm) # Project back to hidden_size if bi-directional
        
        x_trans = self.pos_encoder(x_lstm)
        seq_rep = self.transformer(x_trans)
        seq_rep = self.norm(seq_rep)
        
        # 3. Static Encoding
        dow = self.dow_embed(x_time[:, 0])
        moy = self.moy_embed(x_time[:, 1])
        static_combined = torch.cat([x_stat, dow, moy], dim=1)
        static_emb = self.static_net(static_combined)
        
        # 4. Fusion
        fused_seq = self.cross_att(seq_rep, static_emb)
        weights = F.softmax(self.pool_query(fused_seq), dim=1)
        context = torch.sum(fused_seq * weights, dim=1)
        
        # 5. Heads
        reg_1 = self.head_1_reg(context)
        cls_1 = self.head_1_cls(context) # Note: MoE for Classificaton too in V17
        
        reg_2 = self.head_2_reg(context)
        cls_2 = self.head_2_cls(context)
        
        # Note: RevIN Denormalization for Regression?
        # Usually RevIN denorms the output.
        # But our target is 'log_return', which is stationary-ish.
        # RevIN is most useful for PRICE forecasting.
        # However, if we enabled it, we should arguably denorm the regression output.
        # But since our targets (Y_1, Y_2) are NOT price, but returns, RevIN is mostly acting as a robust scaler for INPUTS.
        # We will NOT denrom output against the input statistics because output domain != input domain.
        
        return reg_1, cls_1, reg_2, cls_2
