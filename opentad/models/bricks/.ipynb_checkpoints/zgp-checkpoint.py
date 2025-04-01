import torch
import torch.nn.functional as F
import torch.nn as nn

from .transformer import AffineDropPath

class ZGPBlock(nn.Module):
    def __init__(self,
                 n_embd,  # dimension of the input features
                 kernel_size=3,  # conv kernel size
                 n_ds_stride=1,  # downsampling stride for the current layer
                 d=1,  # d
                 n_hidden=None,  # hidden dim for ffn
                 path_pdrop=0.0,  # drop path rate
                 act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
                 downsample_type="max",
                 init_conv_vars=1,  # init gaussian variance for the weight):
                 ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        if n_ds_stride > 1:
            if downsample_type == "max":
                kernel_size, stride, padding = n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == "avg":
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                    nn.Conv1d(n_embd, n_embd, 1, 1, 0),
                )
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()

        self.ln = nn.LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        assert int(d) == d

        self.avg_pool = nn.AvgPool1d(3, stride=1, padding=1)
        self.conv_local = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, padding=kernel_size // 2, groups=n_embd)
        self.convdw = nn.Conv1d(n_embd, n_embd, kernel_size, dilation=d, padding=kernel_size // 2, groups=n_embd)
        self.convl = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convg = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)

        if n_hidden is None:
            n_hidden = 4 * n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=1),
            act_layer(),
            nn.Conv1d(n_hidden, n_embd, 1, groups=1),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_ffn = AffineDropPath(n_embd, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_ffn = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.conv_local.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convdw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convl.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convg.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.conv_local.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convdw.bias, 0)
        torch.nn.init.constant_(self.convl.bias, 0)
        torch.nn.init.constant_(self.convg.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        B, C, T = x.shape

        x = self.downsample(x)

        out_mask = F.interpolate(
            mask.unsqueeze(1).to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode="trunc"),
            mode="nearest",
        ).detach()

        out = self.ln(x.permute(0, 2, 1)).permute(0, 2, 1)
        local_avg = self.avg_pool(out)
        local_feature = self.conv_local(out - local_avg)
        local_alpha = self.convl(out) + local_feature
        convw = self.convw(out)
        convdw = self.convdw(out)
        global_alpha = torch.relu(self.convg(out.mean(dim=-1, keepdim=True)))
        fc = self.fc(out)
        out = (convw + convdw) * local_alpha + global_alpha * fc + out
        out = (convw + convdw) * local_alpha + fc + out
        out = x * out_mask + self.drop_path_out(out)

        out = out + self.drop_path_ffn(self.mlp(self.gn(out)))
        return out, out_mask.squeeze(1).bool()