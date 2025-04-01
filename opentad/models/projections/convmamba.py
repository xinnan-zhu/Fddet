import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath
from ..builder import PROJECTIONS

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

@PROJECTIONS.register_module()
class ConvmambaProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
        mamba_kernel_size=4,  # kernel size of causal conv1d in mamba
        channel_expand=2,  # expand ratio for mamba
        num_head=4,  # number of heads in transformer
        drop_path_rate=0.3,
    ):
        super().__init__()
        assert (
            MAMBA_AVAILABLE
        ), "Please install mamba-ssm to use this module. Check: https://github.com/OpenGVLab/video-mamba-suite"

        assert len(arch) == 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.pre_stem = FrequencyLayer(out_channels)
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                HybridCausalBlock(
                    out_channels,
                    stride=1,
                    kernel_size=mamba_kernel_size,
                    expand=channel_expand,
                    num_head=num_head,
                    drop_path_rate=drop_path_rate,
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(
                HybridCausalBlock(
                    out_channels,
                    stride=2,
                    kernel_size=mamba_kernel_size,
                    expand=channel_expand,
                    num_head=num_head,
                    drop_path_rate=drop_path_rate,
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, m):
        # set nn.Linear bias term to 0
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.bias is not None:
                if not getattr(m.bias, "_no_reinit", False):
                    torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        x, mask = self.pre_stem(x, mask)
        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


class HybridCausalBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        stride=1,  # downsampling stride for the current layer
        kernel_size=4,  # conv kernel size
        expand=2,  # expand ratio for mamba
        num_head=4,  # number of heads in transformer
        drop_path_rate=0.3,  # drop path rate
    ):
        super().__init__()

        # normalization
        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        # hybrid block with mamba and self-attn
        self.block = MixtureCausalBlock(n_embd, d_conv=kernel_size, expand=expand)

        # downsampling
        if stride > 1:
            assert stride == 2
            self.downsample = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate, transpose=True)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.block(self.norm(x)))
        x = x.permute(0, 2, 1)
        x = x * mask.unsqueeze(1).to(x.dtype)

        if self.downsample is not None:
            mask = self.downsample(mask.float()).bool()
            x = self.downsample(x) * mask.unsqueeze(1).to(x.dtype)
        return x, mask

class MixtureCausalBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 5, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner * 3, self.d_model, bias=bias)

        self.convs = ConvBranch(in_channels=d_model, dilations=(1, 2, 4))

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # 按比例分割 xz: 4/5 和 1/5
        channels = xz.shape[1]
        split_size1 = int(channels * 4 / 5)  # 第一部分占 4/5
        split_size2 = channels - split_size1  # 第二部分占 1/5
        xz_4, xz_1 = torch.split(xz, [split_size1, split_size2], dim=1)

        xz_f, xz_b = torch.chunk(xz_4, 2, dim=1)  # (B, D, L)
        xz = torch.cat([xz_f, xz_b.flip([-1])], dim=0)

        # causal conv1d -> ssm
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        out = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        out = out.chunk(2)
        out = torch.cat([out[0], out[1].flip([-1])], dim=1)
        out_c = self.convs(xz_1)
        out_f = torch.concat([out, out_c], dim=1)
        out = F.linear(rearrange(out_f, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        return out


class ConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels=None, dilations=(1, 2, 4)):
        super().__init__()
        self.out_channels = in_channels * 2 if out_channels is None else out_channels

        # 局部卷积
        self.local_conv = nn.Conv1d(in_channels, in_channels, 3, padding=1,
                                    groups=in_channels)

        # 全局特征提取部分
        self.global_fc = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)

        # 多尺度卷积前的深度卷积（仅用于 kernel_size > 3）
        self.pre_convs = nn.ModuleDict({
            str(d): nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
            for d in dilations
        })

        # 多尺度空洞卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=d, dilation=d, groups=in_channels)
            for d in dilations
        ])

        # 输出特征的 LayerNorm
        self.fusion_ln = nn.LayerNorm(in_channels * 4)

        # 特征融合层
        self.feature_fusion = nn.Conv1d(in_channels * 4, self.out_channels, kernel_size=1)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 局部卷积初始化
        nn.init.kaiming_normal_(self.local_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.local_conv.bias is not None:
            nn.init.constant_(self.local_conv.bias, 0)

        # 全局特征提取初始化
        nn.init.kaiming_normal_(self.global_fc.weight, mode='fan_out', nonlinearity='relu')
        if self.global_fc.bias is not None:
            nn.init.constant_(self.global_fc.bias, 0)

        # 多尺度前的深度卷积初始化
        for conv in self.pre_convs.values():
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        # 多尺度卷积初始化
        for conv in self.convs:  # 修正为迭代 ModuleList
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)

        # 特征融合层初始化
        nn.init.kaiming_normal_(self.feature_fusion.weight, mode='fan_out', nonlinearity='relu')
        if self.feature_fusion.bias is not None:
            nn.init.constant_(self.feature_fusion.bias, 0)

        # LayerNorm 初始化
        if isinstance(self.fusion_ln, nn.LayerNorm):
            nn.init.constant_(self.fusion_ln.weight, 1.0)
            nn.init.constant_(self.fusion_ln.bias, 0.0)

    def forward(self, xz):
        # 将输入分成两部分
        xz, xz_c = torch.chunk(xz, 2, dim=1)  # 按通道维度分割

        # 局部特征提取
        local_features = F.gelu(self.local_conv(xz_c))  # 使用 GELU 激活函数

        # 全局特征提取
        phi = torch.relu(self.global_fc(xz_c.mean(dim=-1, keepdim=True)))  # 全局特征提取

        # 多尺度特征提取
        features = []
        for conv, pre_conv in zip(self.convs, self.pre_convs.values()):
            x_pre = pre_conv(xz)  # 深度卷积
            features.append(conv(x_pre))
        concat_features = torch.cat(features, dim=1)  # 在通道维度拼接

        # 特征融合
        concat_features = torch.cat([concat_features, phi * local_features], dim=1)  # 使用全局特征进行融合

        # 输出特征的 LayerNorm
        fused_features = rearrange(concat_features, "b c t -> b t c")
        fused_features = self.fusion_ln(fused_features)
        fused_features = rearrange(fused_features, "b t c -> b c t")

        # 特征融合层
        final_features = self.feature_fusion(fused_features)

        return final_features

class FrequencyLayer(nn.Module):
    def __init__(self, d_model):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(0)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        c = 7
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, d_model))
        self.edge = EdgeEnhancer1D(d_model)

    def forward(self, input_tensor, mask):
        # Permute input_tensor to match [batch_size, sequence_length, feature_channel] for FFT operations
        edge_out = self.edge(input_tensor, mask)
        input_tensor = input_tensor.permute(0, 2, 1)  # Now [batch_size, sequence_length, feature_channel]
        batch, seq_len, hidden = input_tensor.shape

        # Apply mask to input_tensor by setting masked elements to zero
        mask = mask.unsqueeze(-1)
        input_tensor = input_tensor * mask

        # FFT operation on the sequence_length dimension
        x_fft = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        # Low-pass filter on FFT result
        low_pass = x_fft[:]
        low_pass[:, self.c:, :] = 0  # Set high frequency components to 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')

        # Compute high-pass component
        high_pass = input_tensor - low_pass

        # Combining low-pass and high-pass components
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        # Apply Dropout
        hidden_states = self.out_dropout(sequence_emb_fft)

        # Apply LayerNorm and residual connection
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # Permute back to original shape [batch_size, feature_channel, sequence_length]
        hidden_states = hidden_states.permute(0, 2, 1)  # [batch_size, feature_channel, sequence_length]

        hidden_states = hidden_states + edge_out

        return hidden_states, mask.squeeze(-1)

class EdgeEnhancer1D(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, in_dim, 1, bias=False)  # 1x1卷积
        self.norm = nn.LayerNorm(in_dim)  # LayerNorm归一化
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.pool = nn.AvgPool1d(3, stride=1, padding=1)  # 平均池化层
        self._initialize_weights()

    def forward(self, x, mask):
        edge = self.pool(x * mask.unsqueeze(1).float())  # 对有效部分进行池化操作
        edge = x - edge  # 计算边缘信息

        # Conv1d
        edge = self.conv(edge)

        # LayerNorm and permute
        edge = edge.permute(0, 2, 1)
        edge = self.norm(edge)
        edge = edge.permute(0, 2, 1)  # 转换回 [batch_size, channels, seq_len]

        edge = self.sigmoid(edge)  # 激活函数
        return x + edge

    def _initialize_weights(self):
        # Xavier initialization for the 1x1 conv layer
        nn.init.xavier_uniform_(self.conv.weight)
