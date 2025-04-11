class DynamicPruner(nn.Module):
    def __init__(self, beta=0.7):
        super().__init__()
        self.beta = beta

    def forward(self, attention_weights):
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        max_entropy = torch.max(entropy)

        # Dynamic mask generation
        keep_mask = (entropy > self.beta * max_entropy).float()
        return keep_mask


# Example of use in UNet
class SparseAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ​ ** -0.5
        self.pruner = DynamicPruner()

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Dynamic pruning
        keep_mask = self.pruner(attn)
        pruned_attn = attn * keep_mask.unsqueeze(-1)

        x = (pruned_attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)