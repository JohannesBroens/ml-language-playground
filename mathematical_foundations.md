# Mathematical Foundations

This document describes the mathematics behind the neural network implementations in this project. All implementations (C, Rust, NumPy, PyTorch) use the same algorithms described here.

## Table of Contents

### Shared Concepts
- [Feature Normalization](#feature-normalization)
- [Activation Functions](#activation-functions)
- [Numerically Stable Softmax](#numerically-stable-softmax)
- [Loss Function](#loss-function)
- [Weight Initialization](#weight-initialization)
- [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)

### Multi-Layer Perceptron (MLP)
- [MLP Architecture](#mlp-architecture)
- [MLP Forward Propagation](#mlp-forward-propagation)
- [MLP Backpropagation](#mlp-backpropagation)
- [MLP Training Algorithm](#mlp-training-algorithm)

### Convolutional Neural Network (CNN — LeNet-5)
- [CNN Architecture](#cnn-architecture)
- [Convolution via im2col](#convolution-via-im2col)
- [Average Pooling](#average-pooling)
- [CNN Forward Propagation](#cnn-forward-propagation)
- [CNN Backpropagation](#cnn-backpropagation)
- [CNN Training Algorithm](#cnn-training-algorithm)

### Reference
- [Notation Reference](#notation-reference)

---

# Shared Concepts

## Feature Normalization

Before training, all input features are z-score normalized per feature:

```math
x'_j = \frac{x_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
```

Where:
- $\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij}$ is the feature mean
- $\sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (x_{ij} - \mu_j)^2$ is the feature variance
- $\epsilon = 10^{-8}$ prevents division by zero for constant features

This centers each feature around zero with unit variance, which helps gradient descent converge faster by ensuring all features are on a similar scale.

---

## Activation Functions

### ReLU (Rectified Linear Unit)

Used in hidden layers of both MLP and CNN:

```math
\text{ReLU}(z) = \max(0, z)
```

Derivative:

```math
\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
```

ReLU avoids the vanishing gradient problem of sigmoid/tanh and is computationally cheap.

### Softmax

Used in the output layer to produce a probability distribution over classes:

```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
```

---

## Numerically Stable Softmax

Direct computation of $e^{z_i}$ can overflow for large logits. The implementation uses the **max subtraction trick**:

```math
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{k} e^{z_j - \max(\mathbf{z})}}
```

This is mathematically equivalent (the constant cancels) but ensures all exponents are $\leq 0$, preventing overflow while preserving numerical precision.

---

## Loss Function

**Cross-entropy loss** for multi-class classification:

```math
L = -\sum_{i=1}^{k} y_i \log(\hat{y}_i + \epsilon)
```

For one-hot encoded labels $\mathbf{y}$ where only the target class $c$ is 1, this simplifies to:

```math
L = -\log(\hat{y}_c + \epsilon)
```

The epsilon ($\epsilon = 10^{-7}$) prevents $\log(0)$.

---

## Weight Initialization

Weights are initialized with **Xavier/He uniform** initialization:

```math
W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{2}{n_{\text{in}}}},\; \sqrt{\frac{2}{n_{\text{in}}}}\right)
```

Where $n_{\text{in}}$ is the fan-in (number of input connections to the layer). For convolutional layers, $n_{\text{in}} = C_{\text{in}} \cdot k_H \cdot k_W$. This scale factor prevents vanishing or exploding activations in networks with ReLU, by keeping the variance of activations approximately constant across layers.

Biases are initialized to zero.

---

## Mini-Batch Gradient Descent

The implementations use **mini-batch SGD**: the training set is divided into batches of size $B$, and weights are updated after processing each batch.

### Full-Batch vs Mini-Batch vs Stochastic GD

| Method | Batch Size | Trade-off |
|--------|-----------|-----------|
| Full-batch GD | $B = N$ | Exact gradient, slow per update, can get stuck |
| Mini-batch SGD | $1 < B < N$ | Good balance of noise and efficiency |
| Stochastic GD | $B = 1$ | Noisy gradient, fast per update, good regularization |

### Gradient Averaging

Per-sample gradients are accumulated over the batch, then averaged by dividing the learning rate:

```math
\eta_{\text{scaled}} = \frac{\eta}{|B|}
```

```math
\theta \leftarrow \theta - \eta_{\text{scaled}} \sum_{i \in B} \nabla_\theta L_i
```

This is equivalent to $\theta \leftarrow \theta - \eta \cdot \frac{1}{|B|}\sum_{i \in B} \nabla_\theta L_i$ but avoids a separate division step.

---

# Multi-Layer Perceptron (MLP)

## MLP Architecture

A single hidden-layer MLP with:

- **Input layer**: $n$ features
- **Hidden layer**: $m$ neurons with ReLU activation
- **Output layer**: $k$ neurons with softmax activation (one per class)

```math
\begin{align*}
\mathbf{h} &= \text{ReLU}\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right) \\
\hat{\mathbf{y}} &= \text{softmax}\left(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\right)
\end{align*}
```

Where:
- $\mathbf{x} \in \mathbb{R}^{n}$: input vector
- $\mathbf{W}^{(1)} \in \mathbb{R}^{m \times n}$: input-to-hidden weights
- $\mathbf{b}^{(1)} \in \mathbb{R}^{m}$: hidden biases (initialized to zero)
- $\mathbf{W}^{(2)} \in \mathbb{R}^{k \times m}$: hidden-to-output weights
- $\mathbf{b}^{(2)} \in \mathbb{R}^{k}$: output biases (initialized to zero)

---

## MLP Forward Propagation

### Hidden Layer

```math
\begin{align*}
\mathbf{z}^{(1)} &= \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)} \\
\mathbf{h} &= \text{ReLU}(\mathbf{z}^{(1)})
\end{align*}
```

### Output Layer

```math
\begin{align*}
\mathbf{z}^{(2)} &= \mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)} \\
\hat{\mathbf{y}} &= \text{softmax}(\mathbf{z}^{(2)})
\end{align*}
```

---

## MLP Backpropagation

### Output Layer Gradient (Softmax + Cross-Entropy Shortcut)

When combining softmax activation with cross-entropy loss, the gradient simplifies elegantly:

```math
\delta^{(2)} = \hat{\mathbf{y}} - \mathbf{y}
```

This avoids computing separate softmax and cross-entropy derivatives.

**Proof**: By the chain rule,
```math
\frac{\partial L}{\partial z_i^{(2)}} = \sum_j \frac{\partial L}{\partial \hat{y}_j} \frac{\partial \hat{y}_j}{\partial z_i^{(2)}}
```
For cross-entropy loss and softmax, using
```math
\frac{\partial \hat{y}_j}{\partial z_i}=\hat{y}_i(\delta_{ij}-\hat{y}_j),
```
this reduces to $\hat{y}_i - y_i$.

### Hidden Layer Gradient (Backprop through ReLU)

```math
\delta^{(1)} = \left((\mathbf{W}^{(2)})^\top \delta^{(2)}\right) \odot \text{ReLU}'(\mathbf{z}^{(1)})
```

Since $\text{ReLU}'(z) = \mathbb{1}[z > 0]$, this zeros out gradients for inactive neurons.

### Weight Gradients

```math
\begin{align*}
\nabla_{\mathbf{W}^{(2)}} L &= \delta^{(2)} \mathbf{h}^\top \\
\nabla_{\mathbf{b}^{(2)}} L &= \delta^{(2)} \\
\nabla_{\mathbf{W}^{(1)}} L &= \delta^{(1)} \mathbf{x}^\top \\
\nabla_{\mathbf{b}^{(1)}} L &= \delta^{(1)}
\end{align*}
```

---

## MLP Training Algorithm

```
Input: dataset {(x_i, y_i)}, epochs T, learning rate η, batch size B
Initialize: W1, b1, W2, b2 with Xavier uniform / zeros

For epoch = 1 to T:
    For each batch B ⊂ {1, ..., N}:
        // Forward pass (per sample in batch)
        z1 = X_B @ W1 + b1
        h  = ReLU(z1)
        z2 = h @ W2 + b2
        ŷ  = stable_softmax(z2)

        // Loss
        L = -mean(log(ŷ[targets] + ε))

        // Backward pass
        δ2 = ŷ - one_hot(targets)           // softmax+CE shortcut
        δ1 = (δ2 @ W2.T) * (h > 0)         // backprop through ReLU

        // Accumulate and average gradients
        η_scaled = η / |B|
        W2 -= η_scaled * (h.T @ δ2)
        b2 -= η_scaled * sum(δ2)
        W1 -= η_scaled * (X_B.T @ δ1)
        b1 -= η_scaled * sum(δ1)
```

---

# Convolutional Neural Network (CNN — LeNet-5)

## CNN Architecture

A LeNet-5 variant for MNIST digit classification (28×28 grayscale images, 10 classes):

```
Input (1×28×28)
  ↓
Conv1 (6 filters, 5×5, stride 1, no padding) + ReLU → 6×24×24
  ↓
AvgPool (2×2) → 6×12×12
  ↓
Conv2 (16 filters, 5×5, stride 1, no padding) + ReLU → 16×8×8
  ↓
AvgPool (2×2) → 16×4×4
  ↓
Flatten → 256
  ↓
FC1 (256→120) + ReLU
  ↓
FC2 (120→84) + ReLU
  ↓
Output (84→10) + Softmax
```

| Layer | Input Shape | Kernel/Units | Output Shape | Parameters |
|-------|-------------|-------------|--------------|------------|
| Conv1 | $1 \times 28 \times 28$ | $6$ filters of $5 \times 5$ | $6 \times 24 \times 24$ | $6 \cdot (1 \cdot 25 + 1) = 156$ |
| Pool1 | $6 \times 24 \times 24$ | $2 \times 2$ average | $6 \times 12 \times 12$ | 0 |
| Conv2 | $6 \times 12 \times 12$ | $16$ filters of $5 \times 5$ | $16 \times 8 \times 8$ | $16 \cdot (6 \cdot 25 + 1) = 2416$ |
| Pool2 | $16 \times 8 \times 8$ | $2 \times 2$ average | $16 \times 4 \times 4$ | 0 |
| FC1 | 256 | 120 | 120 | $256 \cdot 120 + 120 = 30840$ |
| FC2 | 120 | 84 | 84 | $120 \cdot 84 + 84 = 10164$ |
| Output | 84 | 10 | 10 | $84 \cdot 10 + 10 = 850$ |

**Total: 44,426 trainable parameters.**

All data uses NCHW layout: $\text{batch} \times \text{channels} \times \text{height} \times \text{width}$.

---

## Convolution via im2col

Direct convolution requires six nested loops (batch, output channel, output height, output width, input channel, kernel height, kernel width). The **im2col** trick converts convolution into matrix multiplication, enabling reuse of optimized BLAS routines.

### im2col Transform

Given an input $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$, a kernel of size $k_H \times k_W$, stride $s$, and output spatial dimensions $O_H \times O_W$:

```math
O_H = \frac{H - k_H}{s} + 1, \quad O_W = \frac{W - k_W}{s} + 1
```

The im2col operation extracts every $C_{\text{in}} \times k_H \times k_W$ patch from $\mathbf{X}$ and arranges them as columns of a matrix $\mathbf{C} \in \mathbb{R}^{(C_{\text{in}} \cdot k_H \cdot k_W) \times (O_H \cdot O_W)}$:

```math
\mathbf{C}[c \cdot k_H \cdot k_W + p \cdot k_W + q,\; h \cdot O_W + w] = \mathbf{X}[c,\; h \cdot s + p,\; w \cdot s + q]
```

where $c \in [0, C_{\text{in}})$, $p \in [0, k_H)$, $q \in [0, k_W)$, $h \in [0, O_H)$, $w \in [0, O_W)$.

### Convolution as GEMM

With filter weights reshaped as $\mathbf{F} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} \cdot k_H \cdot k_W)}$, convolution becomes:

```math
\mathbf{Y} = \mathbf{F} \cdot \mathbf{C} + \mathbf{b}
```

where $\mathbf{Y} \in \mathbb{R}^{C_{\text{out}} \times (O_H \cdot O_W)}$ and $\mathbf{b} \in \mathbb{R}^{C_{\text{out}}}$ is broadcast across columns.

**Concrete sizes in this network:**

| Layer | $\mathbf{F}$ shape | $\mathbf{C}$ shape | $\mathbf{Y}$ shape |
|-------|----------|----------|----------|
| Conv1 | $6 \times 25$ | $25 \times 576$ | $6 \times 576$ |
| Conv2 | $16 \times 150$ | $150 \times 64$ | $16 \times 64$ |

### col2im (Inverse Transform)

During backpropagation, gradients w.r.t. the column matrix must be scattered back to the input spatial layout. Because im2col duplicates overlapping pixels, col2im **accumulates** (sums) gradients at each original position:

```math
\frac{\partial L}{\partial \mathbf{X}}[c,\; h \cdot s + p,\; w \cdot s + q] \mathrel{+}= \frac{\partial L}{\partial \mathbf{C}}[c \cdot k_H \cdot k_W + p \cdot k_W + q,\; h \cdot O_W + w]
```

---

## Average Pooling

### Forward

For a $p \times p$ pooling window with stride $p$ (non-overlapping):

```math
\mathbf{Y}[c, i, j] = \frac{1}{p^2} \sum_{m=0}^{p-1} \sum_{n=0}^{p-1} \mathbf{X}[c,\; i \cdot p + m,\; j \cdot p + n]
```

Output spatial size: $H_{\text{out}} = H / p$, $W_{\text{out}} = W / p$.

### Backward

Each output gradient distributes equally to all positions in its pooling window:

```math
\frac{\partial L}{\partial \mathbf{X}}[c,\; i \cdot p + m,\; j \cdot p + n] = \frac{1}{p^2} \frac{\partial L}{\partial \mathbf{Y}}[c, i, j]
```

---

## CNN Forward Propagation

The forward pass processes convolutions per-sample (each sample needs its own im2col workspace), then batches the fully connected layers as matrix multiplications.

### Convolutional Layers (per sample)

```math
\begin{align*}
\mathbf{C}_1 &= \text{im2col}(\mathbf{X},\; k=5,\; s=1) & \in \mathbb{R}^{25 \times 576} \\
\mathbf{Z}_1 &= \mathbf{F}_1 \mathbf{C}_1 + \mathbf{b}_1 & \in \mathbb{R}^{6 \times 576} \\
\mathbf{A}_1 &= \text{ReLU}(\mathbf{Z}_1) \\
\mathbf{P}_1 &= \text{AvgPool}_{2 \times 2}(\text{reshape}(\mathbf{A}_1, 6 \times 24 \times 24)) & \in \mathbb{R}^{6 \times 12 \times 12} \\[6pt]
\mathbf{C}_2 &= \text{im2col}(\mathbf{P}_1,\; k=5,\; s=1) & \in \mathbb{R}^{150 \times 64} \\
\mathbf{Z}_2 &= \mathbf{F}_2 \mathbf{C}_2 + \mathbf{b}_2 & \in \mathbb{R}^{16 \times 64} \\
\mathbf{A}_2 &= \text{ReLU}(\mathbf{Z}_2) \\
\mathbf{P}_2 &= \text{AvgPool}_{2 \times 2}(\text{reshape}(\mathbf{A}_2, 16 \times 8 \times 8)) & \in \mathbb{R}^{16 \times 4 \times 4} \\
\mathbf{f} &= \text{flatten}(\mathbf{P}_2) & \in \mathbb{R}^{256}
\end{align*}
```

### Fully Connected Layers (batched)

With $\mathbf{F}_B \in \mathbb{R}^{B \times 256}$ as the batch of flattened features:

```math
\begin{align*}
\mathbf{H}_1 &= \text{ReLU}(\mathbf{F}_B \mathbf{W}_1 + \mathbf{b}^{(1)}) & \in \mathbb{R}^{B \times 120} \\
\mathbf{H}_2 &= \text{ReLU}(\mathbf{H}_1 \mathbf{W}_2 + \mathbf{b}^{(2)}) & \in \mathbb{R}^{B \times 84} \\
\hat{\mathbf{Y}} &= \text{softmax}(\mathbf{H}_2 \mathbf{W}_3 + \mathbf{b}^{(3)}) & \in \mathbb{R}^{B \times 10}
\end{align*}
```

---

## CNN Backpropagation

### FC Layers (batched, same as MLP)

```math
\begin{align*}
\boldsymbol{\delta}_3 &= \hat{\mathbf{Y}} - \mathbf{Y}_{\text{one-hot}} & \text{(softmax + CE shortcut)} \\
\nabla_{\mathbf{W}_3} &= \mathbf{H}_2^\top \boldsymbol{\delta}_3, \quad \nabla_{\mathbf{b}^{(3)}} = \mathbf{1}^\top \boldsymbol{\delta}_3 \\[4pt]
\boldsymbol{\delta}_2 &= (\boldsymbol{\delta}_3 \mathbf{W}_3^\top) \odot \text{ReLU}'(\mathbf{H}_2) \\
\nabla_{\mathbf{W}_2} &= \mathbf{H}_1^\top \boldsymbol{\delta}_2, \quad \nabla_{\mathbf{b}^{(2)}} = \mathbf{1}^\top \boldsymbol{\delta}_2 \\[4pt]
\boldsymbol{\delta}_1 &= (\boldsymbol{\delta}_2 \mathbf{W}_2^\top) \odot \text{ReLU}'(\mathbf{H}_1) \\
\nabla_{\mathbf{W}_1} &= \mathbf{F}_B^\top \boldsymbol{\delta}_1, \quad \nabla_{\mathbf{b}^{(1)}} = \mathbf{1}^\top \boldsymbol{\delta}_1 \\[4pt]
\mathbf{d}_{\text{flat}} &= \boldsymbol{\delta}_1 \mathbf{W}_1^\top & \in \mathbb{R}^{B \times 256}
\end{align*}
```

### Convolutional Layers (per sample, reverse order)

For each sample, reshape $\mathbf{d}_{\text{flat}} \in \mathbb{R}^{256}$ to $16 \times 4 \times 4$:

```math
\begin{align*}
\mathbf{d}_{A_2} &= \text{AvgPool\_backward}(\mathbf{d}_{P_2}) \odot \text{ReLU}'(\mathbf{Z}_2) \\
\nabla_{\mathbf{F}_2} &\mathrel{+}= \mathbf{d}_{A_2} \mathbf{C}_2^\top \\
\nabla_{\mathbf{b}_2} &\mathrel{+}= \text{col\_sum}(\mathbf{d}_{A_2}) \\
\mathbf{d}_{P_1} &= \text{col2im}(\mathbf{F}_2^\top \mathbf{d}_{A_2}) \\[6pt]
\mathbf{d}_{A_1} &= \text{AvgPool\_backward}(\mathbf{d}_{P_1}) \odot \text{ReLU}'(\mathbf{Z}_1) \\
\nabla_{\mathbf{F}_1} &\mathrel{+}= \mathbf{d}_{A_1} \mathbf{C}_1^\top \\
\nabla_{\mathbf{b}_1} &\mathrel{+}= \text{col\_sum}(\mathbf{d}_{A_1})
\end{align*}
```

Gradients accumulate ($\mathrel{+}=$) across all samples in the batch before the SGD update.

### Weight Update

Same mini-batch SGD as MLP:

```math
\theta \leftarrow \theta - \frac{\eta}{|B|} \sum_{i \in B} \nabla_\theta L_i
```

---

## CNN Training Algorithm

```
Input: MNIST {(X_i, y_i)}, epochs T, learning rate η, batch size B
Initialize: F1, b1, F2, b2, W1..W3, b(1)..b(3) with Xavier uniform / zeros

For epoch = 1 to T:
    For each batch B ⊂ {1, ..., N}:

        // Per-sample convolutional forward
        For each sample in batch:
            C1 = im2col(X, k=5, s=1)               // 25 × 576
            Z1 = F1 @ C1 + b1;  A1 = ReLU(Z1)      // 6 × 576
            P1 = avg_pool(A1, 2)                     // 6 × 12 × 12
            C2 = im2col(P1, k=5, s=1)               // 150 × 64
            Z2 = F2 @ C2 + b2;  A2 = ReLU(Z2)      // 16 × 64
            P2 = avg_pool(A2, 2)                     // 16 × 4 × 4
            flat[sample] = flatten(P2)               // 256

        // Batched FC forward
        H1 = ReLU(flat @ W1 + b(1))                 // B × 120
        H2 = ReLU(H1 @ W2 + b(2))                   // B × 84
        Ŷ  = softmax(H2 @ W3 + b(3))                // B × 10

        // Loss
        L = -mean(log(Ŷ[targets] + ε))

        // FC backward
        δ3 = Ŷ - one_hot(targets)
        ∇W3 = H2.T @ δ3;   ∇b3 = sum(δ3)
        δ2 = (δ3 @ W3.T) ⊙ (H2 > 0)
        ∇W2 = H1.T @ δ2;   ∇b2 = sum(δ2)
        δ1 = (δ2 @ W2.T) ⊙ (H1 > 0)
        ∇W1 = flat.T @ δ1;  ∇b1 = sum(δ1)
        d_flat = δ1 @ W1.T

        // Per-sample conv backward
        For each sample in batch:
            d_P2 = reshape(d_flat[sample], 16×4×4)
            d_A2 = pool_backward(d_P2) ⊙ (Z2 > 0)
            ∇F2 += d_A2 @ C2.T;  ∇b2_conv += col_sum(d_A2)
            d_C2 = F2.T @ d_A2;  d_P1 = col2im(d_C2)
            d_A1 = pool_backward(d_P1) ⊙ (Z1 > 0)
            ∇F1 += d_A1 @ C1.T;  ∇b1_conv += col_sum(d_A1)

        // SGD update
        η_s = η / |B|
        All params -= η_s × gradients
```

### Workspace Memory

Each sample requires temporary buffers for im2col and intermediate activations:

| Buffer | Size (floats) | Purpose |
|--------|---------------|---------|
| col1 | $25 \times 576 = 14{,}400$ | im2col for Conv1 |
| conv1_out | $6 \times 576 = 3{,}456$ | Conv1 pre-activation |
| pool1 | $6 \times 144 = 864$ | Pool1 output |
| col2 | $150 \times 64 = 9{,}600$ | im2col for Conv2 |
| conv2_out | $16 \times 64 = 1{,}024$ | Conv2 pre-activation |
| pool2 | 256 | Pool2 output (= flat) |
| **Total** | **29,600** | **~116 KB per sample** |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^{n}$ | Input vector (MLP) or image tensor (CNN) |
| $\mathbf{y} \in \mathbb{R}^{k}$ | True label (one-hot encoded) |
| $\hat{\mathbf{y}} \in \mathbb{R}^{k}$ | Predicted probabilities (softmax output) |
| $\mathbf{h} \in \mathbb{R}^{m}$ | Hidden layer activations |
| $\mathbf{W}, \mathbf{F}$ | Weight matrices (FC layers, conv filters) |
| $\mathbf{b}$ | Bias vectors |
| $\boldsymbol{\delta}$ | Error signals for backpropagation |
| $\mathbf{C}$ | im2col column matrix |
| $\eta$ | Learning rate |
| $B$ | Mini-batch size |
| $\odot$ | Element-wise (Hadamard) product |
| $\mathrel{+}=$ | Accumulate (sum into existing value) |
| $N$ | Total number of training samples |
| $n, m, k$ | Input size, hidden size, number of classes |
| $C_{\text{in}}, C_{\text{out}}$ | Input/output channels (CNN) |
| $k_H, k_W$ | Kernel height and width |
| $O_H, O_W$ | Output spatial dimensions |
| $s$ | Convolution stride |
| $p$ | Pooling window size |

---

# Extended Model Families

The sections below cover the additional model families added on top of the original MLP/CNN core.

## Linear, Ridge, and Lasso Regression

For inputs $X \in \mathbb{R}^{n \times d}$ and targets $y \in \mathbb{R}^n$, augment $X$ with a column of ones to absorb the intercept:

```math
\tilde{X} = [X \ |\ \mathbf{1}], \quad \tilde{w} = [w; b]
```

### Ordinary Least Squares (OLS)

Minimise the residual sum of squares:

```math
\min_{\tilde{w}} \tfrac{1}{2} \| y - \tilde{X}\tilde{w} \|_2^2
```

Setting the gradient to zero yields the normal equations $\tilde{X}^\top \tilde{X}\, \tilde{w} = \tilde{X}^\top y$, solved by Cholesky factorisation $\tilde{X}^\top \tilde{X} = L L^\top$ followed by forward-then-back substitution.

### Ridge

Add an L2 penalty on the weights (but not the bias):

```math
\min_{w, b} \tfrac{1}{2} \| y - X w - b \mathbf{1} \|_2^2 + \tfrac{\lambda}{2} \| w \|_2^2
```

The same Cholesky solve applies after adding $\lambda I$ to the leading $d \times d$ block of $\tilde{X}^\top \tilde{X}$.

### Lasso

Replace the L2 penalty with L1:

```math
\min_w \tfrac{1}{2n} \| y - X w - b \|_2^2 + \lambda \| w \|_1
```

Solved by **cyclic coordinate descent**. For coordinate $j$ define $\rho_j = \frac{1}{n} \mathbf{x}_j^\top (r + \mathbf{x}_j w_j)$ where $r$ is the current residual. The closed-form coordinate update is the **soft-thresholding** operator:

```math
w_j \leftarrow \frac{1}{\| \mathbf{x}_j \|^2 / n} \, S(\rho_j, \lambda), \quad S(z, \lambda) = \mathrm{sign}(z) \max(|z| - \lambda, 0)
```

The bias is updated to the mean of the residual at the end of each pass.

## Quantile Regression

For quantile $\tau \in (0, 1)$, the **pinball loss** is

```math
\rho_\tau(u) = u(\tau - \mathbf{1}\{u < 0\}) = \begin{cases} \tau u & \text{if } u \ge 0, \\ (\tau - 1) u & \text{if } u < 0. \end{cases}
```

For predictions $\hat y_i = w^\top x_i + b$ and residuals $u_i = y_i - \hat y_i$, the (sub)gradient with respect to $\hat y_i$ is

```math
\frac{\partial \rho_\tau(u_i)}{\partial \hat y_i} = \mathbf{1}\{\hat y_i > y_i\} - \tau,
```

which is $1 - \tau$ when over-predicting and $-\tau$ when under-predicting. Several quantiles can be trained simultaneously by stacking weight vectors and using the corresponding $\tau$ vector in the gradient.

## CART Regression Tree

Build a binary tree by greedy minimisation of the **weighted squared-error** at each split. For a node containing samples with weights $w_i$ and targets $y_i$, the within-node SSE is

```math
\mathrm{SSE} = \sum_i w_i y_i^2 - \frac{(\sum_i w_i y_i)^2}{\sum_i w_i}.
```

A split on feature $j$ at threshold $t$ partitions the samples into left and right child sets and produces SSE-reduction $\Delta = \mathrm{SSE}_\text{parent} - \mathrm{SSE}_L - \mathrm{SSE}_R$. The best split maximises $\Delta$. Cumulative-sum tricks make the search $O(n \log n)$ per feature.

## Random Forest

Fit $T$ trees on bootstrap samples of the training data; at each split consider only a random subset (e.g. $\sqrt{d}$) of the features. Predictions are averaged across the ensemble:

```math
\hat y(x) = \frac{1}{T} \sum_{t=1}^T f_t(x).
```

The decorrelation between trees, induced by both bagging and feature subsampling, reduces ensemble variance without increasing bias.

## Gradient Boosted Trees

Build a sequence of trees $f_1, f_2, \dots, f_T$ that progressively reduce the loss:

```math
F_m(x) = F_{m-1}(x) + \eta f_m(x).
```

Each tree fits the negative gradient of the loss with respect to the current prediction:

- **Squared error**: $-\partial L / \partial F = y - F$, so $f_m$ fits the current residual.
- **Pinball loss** for quantile $\tau$: $-\partial L / \partial F = \mathbf{1}\{y > F\} \tau - \mathbf{1}\{y \le F\}(1 - \tau)$, which is $\tau$ when under-predicting and $\tau - 1$ when over-predicting.

Stochastic boosting samples a fraction `subsample` of the data for each tree; together with shrinkage (`learning_rate` $< 1$) this is the most reliable way to reduce overfitting in deep boosted ensembles.

## Gaussian Process Regression

For a kernel $k(\cdot, \cdot)$ (here the squared-exponential plus white noise) and training data $(X, y)$, define $K = k(X, X) + \sigma_n^2 I$. The posterior at test points $X_*$ is Gaussian with

```math
\begin{aligned}
\mathbf{m}_* &= k(X_*, X) K^{-1} y, \\
\Sigma_*    &= k(X_*, X_*) - k(X_*, X) K^{-1} k(X, X_*).
\end{aligned}
```

Stable computation factorises $K = L L^\top$ via Cholesky, computes $\alpha = L^{-\top}(L^{-1} y)$, and predicts $\mathbf{m}_* = k(X_*, X) \alpha$. The variance uses $\mathbf{v} = L^{-1} k(X, X_*)$ so $\sigma^2_* = k(x_*, x_*) - \mathbf{v}^\top \mathbf{v}$.

Hyperparameters $\theta = (\ell, \sigma_f, \sigma_n)$ are fit by maximising the **log marginal likelihood**

```math
\log p(y \mid X, \theta) = -\tfrac{1}{2} y^\top \alpha - \sum_i \log L_{ii} - \tfrac{n}{2} \log(2\pi).
```

## Hidden Markov Model (Gaussian emissions)

Hidden states $z_t \in \{1, \dots, K\}$, observations $x_t \in \mathbb{R}$, parameters $\theta = (\pi, T, \mu, \sigma)$ where $T_{ij} = P(z_{t+1} = j \mid z_t = i)$ and emissions are $\mathcal{N}(\mu_k, \sigma_k^2)$.

### Forward / backward in log-space

```math
\log \alpha_t(k) = \log p(x_t \mid z_t = k) + \log \sum_j \exp(\log \alpha_{t-1}(j) + \log T_{jk}).
```

Both directions are computed with log-sum-exp for numerical stability.

### Baum–Welch (EM)

E-step: posteriors $\gamma_t(k) \propto \alpha_t(k) \beta_t(k)$ and pairwise $\xi_t(i, j) \propto \alpha_t(i) T_{ij} p(x_{t+1} \mid j) \beta_{t+1}(j)$.

M-step:

```math
\pi_k = \gamma_1(k), \quad T_{ij} = \frac{\sum_t \xi_t(i, j)}{\sum_{j'} \sum_t \xi_t(i, j')}, \quad \mu_k = \frac{\sum_t \gamma_t(k) x_t}{\sum_t \gamma_t(k)}, \quad \sigma_k^2 = \frac{\sum_t \gamma_t(k) (x_t - \mu_k)^2}{\sum_t \gamma_t(k)}.
```

### Viterbi decoding

Replace the inner sum in the forward pass with a max to obtain the most likely state sequence.

## LSTM and GRU

### LSTM cell

```math
\begin{aligned}
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \\
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \\
\tilde c_t &= \tanh(W_c x_t + U_c h_{t-1} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
```

### GRU cell

```math
\begin{aligned}
r_t &= \sigma(W_r x_t + U_r h_{t-1} + b_r) \\
z_t &= \sigma(W_z x_t + U_z h_{t-1} + b_z) \\
\tilde h_t &= \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h) \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t
\end{aligned}
```

The GRU collapses the input/forget split into a single update gate $z_t$ and merges the cell state with the hidden state, saving roughly 25% of the LSTM's parameters.

## Temporal Convolutional Network (TCN)

A causal dilated 1D convolution applies the kernel only to past timesteps:

```math
y_t = \sum_{k=0}^{K-1} W_k\, x_{t - d \cdot k},
```

where $d$ is the dilation factor and the input is left-padded with $(K - 1) d$ zeros. Stacking $L$ blocks with dilations $1, 2, 4, \dots, 2^{L-1}$ gives an exponential receptive field

```math
r = 1 + 2 (K - 1) (2^L - 1).
```

Each TCN block in the implementation contains two such convolutions, each followed by LayerNorm, ReLU, and dropout, with a residual connection projecting the input channels to the output channels via a $1 \times 1$ convolution when needed.

## Multi-Head Self-Attention

For queries $Q$, keys $K$, and values $V$ in $\mathbb{R}^{T \times d_\text{head}}$,

```math
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_\text{head}}}\right) V.
```

In multi-head attention the model projects $Q, K, V$ into $H$ separate heads, runs the attention per head, and concatenates the outputs through a final linear projection. A causal mask sets the lower-triangular complement to $-\infty$ so position $i$ cannot attend to $j > i$.

## Temporal Fusion Transformer

Three core building blocks plus four assembled stages.

### Gated Linear Unit (GLU)

```math
\mathrm{GLU}(x) = (W x + b) \odot \sigma(V x + c).
```

### Gated Residual Network (GRN)

With optional context $c$,

```math
\begin{aligned}
\eta_2 &= \mathrm{ELU}(W_1 x + W_2 c + b_1) \\
\eta_1 &= W_3 \eta_2 + b_2 \\
\mathrm{GRN}(x, c) &= \mathrm{LayerNorm}(\mathrm{skip}(x) + \mathrm{GLU}(\mathrm{Dropout}(\eta_1))).
\end{aligned}
```

The skip connection projects $x$ to the output dimension when needed.

### Variable Selection Network (VSN)

For input variables $x^{(1)}, \dots, x^{(F)} \in \mathbb{R}$ at each timestep:

1. Embed each variable independently: $e^{(j)} = \mathrm{GRN}_j(x^{(j)}) \in \mathbb{R}^d$.
2. Compute selection logits over the flattened concatenation $[e^{(1)} \| \cdots \| e^{(F)}]$ via a context-aware GRN, then softmax to get weights $\alpha \in \Delta^{F-1}$.
3. Output $\sum_j \alpha_j e^{(j)}$.

The softmax weights $\alpha$ are interpretable as the model's per-timestep importance over inputs.

### Static contexts

A separate VSN encodes the static feature vector. Four GRNs map that single static embedding into four context vectors:

```math
c_\text{vs}, \ c_\text{e}, \ c_\text{h}, \ c_\text{c}
```

used for variable selection ($c_\text{vs}$), static enrichment ($c_\text{e}$), and the LSTM initial $(h_0, c_0)$ states.

### Locality-aware seq2seq

```math
(\text{enc out}, h_T, c_T) = \mathrm{LSTM}_\text{enc}(\text{past embeddings},\ (c_\text{h}, c_\text{c})),\quad \text{dec out} = \mathrm{LSTM}_\text{dec}(\text{future embeddings},\ (h_T, c_T)).
```

The concatenated $[\text{enc out} \| \text{dec out}]$ is fused with the variable-selected embeddings via a GLU + residual + LayerNorm.

### Static enrichment + interpretable attention

```math
\phi_t = \mathrm{GRN}(\text{lstm}_t,\ c_\text{e}), \quad \mathrm{Attn}_\text{interp}(\phi) = \mathrm{Out}\!\left(\frac{1}{H} \sum_{h} \mathrm{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_\text{head}}}\right)\, V_\text{shared}\right).
```

Because all heads share the same value projection $V_\text{shared}$, the head-averaged attention weights are directly interpretable as importance per past timestep.

### Position-wise feed-forward + quantile head

A second GRN per timestep (gated and residual-summed), followed by a linear projection to $Q$ quantiles. Training minimises the mean pinball loss over horizon and quantiles.

## Autoencoder

A symmetric MLP encoder $f_\theta : \mathbb{R}^d \to \mathbb{R}^k$ and decoder $g_\phi : \mathbb{R}^k \to \mathbb{R}^d$ trained to minimise reconstruction error

```math
\mathcal{L}(\theta, \phi) = \tfrac{1}{n} \sum_i \| g_\phi(f_\theta(x_i)) - x_i \|_2^2.
```

Unusual inputs incur a larger reconstruction error; flagging the top quantile yields an unsupervised anomaly score.

## K-Means and GMM

### K-Means (Lloyd's algorithm with k-means++ init)

Alternate **assign** ($z_i = \arg\min_k \| x_i - \mu_k \|_2^2$) and **update** ($\mu_k = \mathrm{mean}\{ x_i : z_i = k \}$) until centroids stop moving. The k-means++ initialisation seeds centres proportional to squared distance from the current set, which gives better convergence guarantees than uniform sampling.

### Gaussian Mixture Model (EM, diagonal covariance)

E-step computes responsibilities

```math
r_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}.
```

M-step updates

```math
\pi_k = \frac{1}{n}\sum_i r_{ik}, \quad \mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}, \quad \Sigma_k = \frac{\sum_i r_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i r_{ik}}.
```

The implementation here uses a diagonal covariance for stability; the same EM framework extends straightforwardly to full or shared covariance.

## PCA

Centre $X$ and take the thin SVD

```math
X - \bar X = U \Sigma V^\top.
```

The first $k$ rows of $V^\top$ are the orthonormal principal components; $\Sigma_k^2 / \mathrm{tr}(\Sigma^2)$ gives the cumulative explained-variance ratio. Reconstruction is $\hat X = (X - \bar X) V_k^\top V_k + \bar X$, and reconstruction MSE is the standard reconstruction-quality metric.
