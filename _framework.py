import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.cuda.amp import autocast, GradScaler

# --------------------------
# 1. Muon Optimizer
# --------------------------
class MuonOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9):
        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Orthogonal gradient projection (simplified Gram-Schmidt)
                grad = p.grad.data
                state = self.state[p]
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)
                
                state['momentum'].mul_(group['beta']).add_(grad, alpha=1 - group['beta'])
                
                # Orthogonalization (Newton-Schultz iteration is used in the actual paper)
                if len(state) > 1:  
                    prev_grad = state['prev_grad']
                    dot_product = torch.sum(state['momentum'] * prev_grad)
                    grad_proj = dot_product / torch.sum(prev_grad**2) * prev_grad
                    state['momentum'].sub_(grad_proj)
                
                # 参数更新
                p.data.add_(state['momentum'], alpha=-group['lr'])
                state['prev_grad'] = state['momentum'].clone()

# --------------------------
# 2. Dynamic pruning attention layer
# --------------------------
class DynamicSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, pruning_ratio=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.pruning_ratio = pruning_ratio
        
        # Standard long attention
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def compute_entropy(self, x):
        # Entropy is calculated as an importance score
        prob = F.softmax(x, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1)
        return entropy.mean(dim=1)  # The entropy of the average head

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        
        # Calculate the attention score
        attn = (q @ k.transpose(-2, -1)) * (C ​** -0.5)
        
        # Dynamic pruning: selection of important heads based on entropy
        entropy = self.compute_entropy(attn)
        k = int(self.num_heads * (1 - self.pruning_ratio))
        _, topk_indices = torch.topk(entropy, k, dim=1)
        
        # Apply sparsity
        selected_attn = torch.gather(attn, 1, topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, N, N))
        attn = torch.zeros_like(attn).scatter(1, topk_indices.unsqueeze(-1).unsqueeze(-1), selected_attn)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

# --------------------------
# 3. Lightweight UNet structure (simplified version)
# --------------------------
class LightweightUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_dim=32):
        super().__init__()
        # Encoder (Dynamic Pruning Attention)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, padding=1),
            DynamicSparseAttention(base_dim, num_heads=4)
        )
        
        # Intermediate Block (Depth Separable Convolution)
        self.mid = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, 3, padding=1),
            nn.Conv2d(base_dim*2, base_dim*2, 3, groups=base_dim*2, padding=1)
        )
        
        # Decoder (Mixed Precision)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, 3, padding=1),
            nn.GroupNorm(4, base_dim)
        )
        self.final = nn.Conv2d(base_dim, out_ch, 1)
    
    @autocast()  # Automatic Mixed Precision
    def forward(self, x):
        x = self.enc1(x)
        x = self.mid(x)
        x = self.dec1(x)
        return self.final(x)

# --------------------------
# 4. Course Learning Controller
# --------------------------
class CurriculumController:
    def __init__(self, total_epochs):
        self.epoch = 0
        self.total = total_epochs
        self.phase = 1
    
    def adjust_hyperparams(self, model, optimizer):
        self.epoch += 1
        
        # Phased adjustments
        if self.epoch < self.total * 0.3:  # Phase 1
            for layer in model.modules():
                if isinstance(layer, DynamicSparseAttention):
                    layer.pruning_ratio = 0.2  # 低剪枝率
        elif self.epoch < self.total * 0.7:  # Phase 2
            for layer in model.modules():
                if isinstance(layer, DynamicSparseAttention):
                    layer.pruning_ratio = 0.5
        else:  # Phase 3
            for layer in model.modules():
                if isinstance(layer, DynamicSparseAttention):
                    layer.pruning_ratio = 0.7

# --------------------------
# An example of a training process
# --------------------------
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the model and optimizer
    model = LightweightUNet().to(device)
    optimizer = MuonOptimizer(model.parameters(), lr=1e-3)
    scaler = GradScaler()  # Mixed-precision scaling
    
    # Course Controller
    controller = CurriculumController(total_epochs=100)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Simulate data loading
    from torch.utils.data import DataLoader
    dataset = torch.rand(100, 3, 128, 128) 
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(100):
        for batch in loader:
            x = batch.to(device)
            
            # Mixed-precision forward
            with autocast():
                pred = model(x)
                loss = criterion(pred, x)  # Autoencoder reconstruction task
                
            # Muon optimizer backpropagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # Adjust your curriculum strategy
        controller.adjust_hyperparams(model, optimizer)
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
