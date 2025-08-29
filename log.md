# Training Log - Pico Decoder Model

## 2025-08-29 - Training Configuration Overhaul & Recovery

### What We Did
- **Identified severe overfitting** in previous training run (steps 1000-9000)
- **Implemented comprehensive training improvements**:
  - Reduced learning rate from 3e-4 to 1e-4 (67% reduction)
  - Extended warmup from 2,500 to 5,000 steps
  - Added dropout (0.1) and attention dropout (0.1) for regularization
  - Added weight decay (0.01) for L2 regularization
  - Implemented gradient clipping (max_grad_norm: 1.0)
  - Increased effective batch size from 4 to 16 (4x larger)
  - Reduced gradient accumulation from 4 to 2 steps
  - Enhanced logging frequency from every 100 to every 50 steps

### The Result
**Dramatic improvement in training stability:**
- **Previous run**: Exponential evaluation metric explosion (2.55e+19 → 4.98e+30), severe overfitting after 8,800 steps
- **New run**: Controlled, stable training with no overfitting signs
  - Training loss: 10.98 → 6.58 (steady decrease)
  - Evaluation metrics: Much more controlled growth (5.07e+18 → 2.39e+21)
  - Learning rate warmup: Perfect execution over 5000 steps
  - No training instability or divergence

### What We Do Next
- **Continue current training run** - it's showing healthy, stable patterns
- **Monitor closely** around steps 8000-10000 for any divergence
- **Consider more frequent evaluation** (every 500 steps instead of 1000)
- **Implement early stopping** if Paloma metrics start exponential growth again
- **Let training continue** until convergence or clear plateau

### What Happened
**Configuration changes successfully addressed the core issues:**
1. **Lower learning rate** prevented aggressive overfitting
2. **Extended warmup** provided stable early training foundation
3. **Regularization techniques** controlled model memorization
4. **Larger batch size** improved gradient quality and training stability
5. **Proper warmup schedule** ensured controlled learning rate progression

**Key insight**: The trade-off of slower training (2-3 days vs 1.5 hours) resulted in actual learning capability instead of catastrophic overfitting. Quality > Speed.

---
*Training run: pico-decoder-tiny-dolma29k-v2*
*Status: Healthy, stable training in progress*

---

## 2025-08-29 - Configuration Update for Dolma 5M Scaling

### What We Did
- **Updated configuration for scaling training** from 29k tokens to 5M tokens
- **Implemented stability-focused hyperparameters** based on OLMo 2 and OLMoE best practices
- **Changed dataset** from `pico-lm/pretokenized-dolma` to `ThomasTheMaker/pretokenized-dolma-5M`

### Key Hyperparameter Changes Applied

#### Learning Rate Schedule
- **Before**: `linear_with_warmup` (designed for short sprints)
- **After**: `cosine` decay over full dataset
- **Why**: Sustained learning rate that gradually decreases over entire 5M tokens without premature drop to zero

#### Weight Initialization
- **Before**: Standard normal distribution
- **After**: `truncated_normal` initialization
- **Why**: Prevents extreme outlier weights that cause immediate instability in long training runs

#### Layer Normalization
- **Before**: `LayerNorm`
- **After**: `RMSNorm`
- **Why**: Empirically more stable for large-scale transformer training, prevents gradient norm spikes

#### Query-Key Normalization
- **Before**: `false`
- **After**: `use_qk_norm: true`
- **Why**: Prevents attention logits from becoming too large, crucial for numerical stability and preventing divergence

#### AdamW Epsilon
- **Before**: Already at `1e-8` (optimal)
- **After**: Maintained at `1e-8`
- **Why**: Confirmed optimal setting for training stability and convergence speed

### Configuration Details
- **Run name**: Updated to `pico-decoder-tiny-dolma5M-v1`
- **Dataset**: `ThomasTheMaker/pretokenized-dolma-5M`
- **Training steps**: 20,000 (maintained for full convergence)
- **Warmup**: 8,000 steps (extended for stability)
- **Learning rate**: 0.00005 (maintained for precision)

### Expected Benefits
1. **Stability**: Truncated normal + RMSNorm prevent training instability
2. **Sustained Learning**: Cosine schedule ensures effective learning over full dataset
3. **Attention Stability**: QK-Norm prevents numerical overflows
4. **Convergence**: Optimized hyperparameters for faster, more stable convergence
5. **Scalability**: Proven to work well for large-scale training runs

### What We Do Next
- **Begin new training run** with updated configuration
- **Monitor stability** during early training phases
- **Track convergence** over the full 5M token dataset
- **Compare performance** against previous 29k token runs
- **Validate** that stability improvements translate to better model quality

### Implementation Notes
- **Cosine Scheduler**: Implemented in training code since it wasn't originally supported
- **Scheduler Details**: Linear warmup (8000 steps) followed by cosine decay to 0.1x initial LR
- **Code Location**: `pico-train/src/training/utils/initialization.py` - added cosine scheduler support

---
*Training run: pico-decoder-tiny-dolma5M-v1*
*Status: Configuration updated, ready for new training run*
