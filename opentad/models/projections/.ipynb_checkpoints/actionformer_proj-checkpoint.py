import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, TransformerBlock, TransCNN
from ..builder import PROJECTIONS


@PROJECTIONS.register_module()
class Conv1DTransformerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        attn_cfg=None,  # n_head n_mha_win_size attn_pdrop
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.proj_pdrop = conv_cfg["proj_pdrop"]
        self.scale_factor = 2  # as default
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0  # as default
        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

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
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

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


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


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
