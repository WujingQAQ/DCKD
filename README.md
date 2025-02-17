# DCKD
# Dynamic Curriculum Knowledge Distillation (DCKD)

This repository implements Dynamic Curriculum Knowledge Distillation with Adaptive Weight Strategy, which combines homoscedastic uncertainty-based adaptive weighting and curriculum learning for knowledge distillation.

## Method

### 1. Adaptive Weight Distillation (AWD)
AWD extends traditional deep learning by incorporating data noise estimation with Gaussian noise distribution. The optimization objective is:
```math
L(\theta_{stu},\sigma_1,\sigma_2) = \min_{\theta_{stu},\sigma_1,\sigma_2} \sum_{x \in \mathcal{D}} L_{ce}(f^s(x; \theta_{stu}), GT, \sigma_1) + L_{kd}(f^t(x; \theta_{tea}), f^s(x; \theta_{stu}), \sigma_2)
```

Where:
- σ₁: Task noise between student network and ground truth
- σ₂: Task noise between student and teacher networks
- L_{ce}: Cross-entropy loss
- L_{kd}: Knowledge distillation loss

### 2. Crescendo Adversarial Distillation (CAD)
CAD implements a curriculum learning strategy that progressively increases difficulty through:

```math
\lambda(\text{epoch}) = \lambda_{\text{min}} + \frac{\lambda_{\text{max}} - \lambda_{\text{min}}}{1 + e^{-k(\text{epoch} - x_0)}}
```

Where:
- λ_min, λ_max: Range of curriculum difficulty
- k: Growth rate parameter
- x₀: Adversarial point (hotspot of distillation)

## Implementation

### Configuration
```yaml
DCKD:
    TEMPERATURE: 4.0      # Temperature for soft targets
    INIT_SIGMA: 1.0       # Initial σ value
    MIN_SIGMA: 0.1        # Minimum constraint for σ
    LAMBDA_MIN: 0.0       # Minimum curriculum difficulty
    LAMBDA_MAX: 1.0       # Maximum curriculum difficulty
    K: 0.1               # Curriculum growth rate
    X0: 50               # Adversarial point
```

### Key Components

1. **Uncertainty Parameters**:
```python
self.log_sigma1 = nn.Parameter(torch.ones(1) * np.log(self.init_sigma))  # CE loss
self.log_sigma2 = nn.Parameter(torch.ones(1) * np.log(self.init_sigma))  # KD loss
self.log_sigma3 = nn.Parameter(torch.ones(1) * np.log(self.init_sigma))  # Feature loss
```

2. **Loss Computation**:
```python
weighted_loss_ce = loss_ce / (2 * sigma1**2)
weighted_loss_kd = loss_kd / (2 * sigma2**2) * lambda_curr
weighted_loss_feat = loss_feat / (2 * sigma3**2) * lambda_curr
```

3. **Gradient Updates**:
```python
# For student parameters
θ_stu ← θ_stu - μ ∂L/∂θ_stu

# For uncertainty parameters
σ₁ ← σ₁ - μ ∂L/∂σ₁  # Minimize CE loss
σ₂ ← σ₂ + μ ∂L/∂σ₂  # Maximize KD loss
```

## Usage

### Training Example
```python
from mdistiller.distillers.DCKD import DCKD

# Initialize models and distiller
distiller = DCKD(student_model, teacher_model, cfg)
optimizer = DistillationOrientedTrainer(
    distiller.get_learnable_parameters(),
    lr=cfg.SOLVER.LR,
    momentum=cfg.DCKD.MOMENTUM,
    momentum_kd=cfg.DCKD.MOMENTUM_KD
)

# Training loop
for epoch in range(cfg.SOLVER.EPOCHS):
    # Forward pass
    logits_student, losses_dict = distiller(
        image=batch_images,
        target=batch_labels,
        epoch=epoch
    )
    
    # Two-step optimization
    optimizer.zero_grad()
    losses_dict['loss_kd'].backward(retain_graph=True)
    optimizer.step_kd()
    
    optimizer.zero_grad()
    losses_dict['loss_ce'].backward()
    optimizer.step()
```

## Monitoring

The implementation provides comprehensive monitoring through losses_dict:
```python
losses_dict = {
    "loss_ce": weighted_loss_ce,        # CE loss
    "loss_kd": weighted_loss_kd,        # KD loss
    "loss_feat": weighted_loss_feat,    # Feature loss
    "loss_reg": reg_term,               # Regularization term
    "sigma1": sigma1,                   # CE uncertainty
    "sigma2": sigma2,                   # KD uncertainty
    "sigma3": sigma3,                   # Feature uncertainty
    "lambda": lambda_curr,              # Current curriculum difficulty
}
```

## Requirements
- PyTorch >= 1.7.0
- tensorboardX
- yacs
- numpy
- tqdm

## License
This project is released under the MIT license.

 
