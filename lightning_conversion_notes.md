# PyTorch Lightning Conversion Guide for ManiFlow

**Status:** Planning - Get baseline training run first, then convert

## Prerequisites

Before converting, ensure you have:
- [ ] A successful training run with current implementation
- [ ] Saved checkpoints and metrics from baseline run
- [ ] WandB logs or equivalent for comparison
- [ ] Note the final validation loss and any key metrics

## Current Architecture Overview

### Training Flow (Current)
Located in: `ManiFlow/maniflow/workspace/train_maniflow_robotwin_workspace.py`

```
TrainManiFlowRoboTwinWorkspace.run()
├── Initialize model, optimizer, dataloaders
├── Manual training loop (lines 208-276)
│   ├── Batch iteration with tqdm
│   ├── Manual device transfer
│   ├── model.compute_loss(batch, ema_model)
│   ├── Manual backward pass
│   ├── Manual optimizer step with gradient accumulation
│   └── Manual EMA update
├── Validation loop (lines 311-330)
├── Diffusion sampling on training batch (lines 332-349)
└── Checkpointing (lines 354-386)
```

### Key Components to Preserve
1. **EMA Model** - Passed to `compute_loss(batch, ema_model)`
2. **Gradient Accumulation** - Manual conditional optimizer step
3. **Diffusion Sampling** - Periodic sampling on saved training batch
4. **TopK Checkpoint Management** - Custom `TopKCheckpointManager`
5. **Hydra Config Integration** - `hydra.utils.instantiate()`

## Conversion Plan

### Phase 1: Setup (Estimated: 30 min)

#### 1.1 Install PyTorch Lightning
```bash
pip install pytorch-lightning
# or add to requirements.txt
```

#### 1.2 Create New File
Create: `ManiFlow/maniflow/workspace/train_maniflow_robotwin_lightning.py`

Keep the old file for comparison!

### Phase 2: Create LightningModule (Estimated: 1-2 hours)

#### 2.1 Basic LightningModule Structure

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from typing import Dict, Any

class ManiFlowLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for ManiFlow policy training.
    
    Key differences from manual training:
    - Lightning handles device management automatically
    - training_step replaces manual training loop
    - validation_step replaces manual validation loop
    - configure_optimizers returns optimizer and scheduler
    """
    
    def __init__(self, cfg: OmegaConf):
        super().__init__()
        # Save all hyperparameters
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        # Initialize policy model
        self.model = hydra.utils.instantiate(cfg.policy)
        
        # For diffusion sampling
        self.train_sampling_batch = None
        self.ema_model = None  # Will be set by callback
        
    def forward(self, batch):
        """Forward pass - not typically used in training."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """
        Replaces lines 216-257 in original code.
        Lightning handles:
        - Device transfer (automatic)
        - Backward pass (automatic, unless manual optimization)
        - Optimizer step (automatic, respects accumulate_grad_batches)
        - Progress bar (via self.log)
        """
        # Save first batch for diffusion sampling
        if self.train_sampling_batch is None:
            self.train_sampling_batch = {k: v.detach().clone() for k, v in batch.items()}
        
        # Get EMA model from callback (see EMA callback section below)
        ema_model = self._get_ema_model()
        
        # Compute loss (same as original)
        loss, loss_dict = self.model.compute_loss(batch, ema_model)
        
        # Log everything
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(loss_dict, on_step=True, on_epoch=False)
        
        # Log learning rate
        opt = self.optimizers()
        self.log('lr', opt.param_groups[0]['lr'], on_step=True, on_epoch=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Replaces lines 317-326 in original code.
        """
        ema_model = self._get_ema_model()
        loss, loss_dict = self.model.compute_loss(batch, ema_model)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({f'val_{k}': v for k, v in loss_dict.items()}, 
                      on_step=False, on_epoch=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """
        Replaces lines 332-349 in original code.
        Runs diffusion sampling on saved training batch.
        """
        if self.current_epoch % self.cfg.training.sample_every != 0:
            return
            
        if self.train_sampling_batch is None:
            return
        
        # Get policy (use EMA if available)
        policy = self.ema_model if self.ema_model is not None else self.model
        policy.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in self.train_sampling_batch.items()}
            obs_dict = batch['obs']
            gt_action = batch['action']
            
            # Run prediction
            result = policy.predict_action(obs_dict)
            pred_action = result['action_pred']
            
            # Compute MSE
            mse = F.mse_loss(pred_action, gt_action)
            self.log('train_action_mse_error', mse)
        
        policy.train()
    
    def configure_optimizers(self):
        """
        Replaces lines 84-86 and scheduler setup in original code.
        Returns optimizer and scheduler configuration.
        """
        # Create optimizer
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, 
            params=self.model.parameters()
        )
        
        # Create scheduler
        from maniflow.model.common.lr_scheduler import get_scheduler
        
        scheduler = get_scheduler(
            optimizer,
            name=self.cfg.training.lr_scheduler,
            num_warmup_steps=self.cfg.training.lr_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step, not epoch
                'frequency': 1,
            }
        }
    
    def _get_ema_model(self):
        """Helper to get EMA model from callback."""
        if not self.cfg.training.use_ema:
            return None
            
        for callback in self.trainer.callbacks:
            if isinstance(callback, EMACallback):
                return callback.ema_model
        return None
```

### Phase 3: Create EMA Callback (Estimated: 30 min)

The tricky part: your `compute_loss()` needs the EMA model as an argument.

```python
class EMACallback(Callback):
    """
    Exponential Moving Average callback.
    Maintains EMA copy of model and makes it available to training_step.
    
    Replaces lines 242-243 in original code.
    """
    
    def __init__(self, decay=0.999, use_ema=True):
        super().__init__()
        self.decay = decay
        self.use_ema = use_ema
        self.ema_model = None
        
    def on_fit_start(self, trainer, pl_module):
        """Initialize EMA model."""
        if not self.use_ema:
            return
            
        import copy
        try:
            self.ema_model = copy.deepcopy(pl_module.model)
        except:
            # Some models (e.g., with Minkowski engine) can't be deepcopied
            # Recreate from config
            self.ema_model = hydra.utils.instantiate(pl_module.cfg.policy)
        
        self.ema_model.to(pl_module.device)
        self.ema_model.eval()
        
        # Store reference in pl_module for easy access
        pl_module.ema_model = self.ema_model
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update EMA model after each training batch."""
        if not self.use_ema or self.ema_model is None:
            return
            
        # Update EMA parameters
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(), 
                pl_module.model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1 - self.decay
                )
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Ensure EMA model is on correct device."""
        if self.ema_model is not None:
            self.ema_model.to(pl_module.device)
```

### Phase 4: Setup Trainer (Estimated: 30 min)

```python
def train_lightning(cfg: OmegaConf, output_dir: str):
    """
    Main training function using PyTorch Lightning.
    Replaces TrainManiFlowRoboTwinWorkspace.run()
    """
    
    # Set seeds
    pl.seed_everything(cfg.training.seed)
    
    # Create datasets and dataloaders
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0
    )
    
    val_dataset = hydra.utils.instantiate(cfg.task.val_dataset)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=cfg.training.num_workers > 0
    )
    
    # Create Lightning module
    model = ManiFlowLightningModule(cfg)
    
    # Setup WandB logger
    wandb_logger = WandbLogger(
        project='maniflow',
        name=cfg.task.name,
        save_dir=output_dir,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Setup callbacks
    callbacks = []
    
    # 1. EMA callback
    ema_callback = EMACallback(
        decay=cfg.training.ema_decay if cfg.training.use_ema else None,
        use_ema=cfg.training.use_ema
    )
    callbacks.append(ema_callback)
    
    # 2. Checkpoint callback (replaces TopKCheckpointManager)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='epoch={epoch:04d}-val_loss={val_loss:.4f}',
        monitor=cfg.checkpoint.topk.monitor_key,  # e.g., 'val_loss'
        mode=cfg.checkpoint.topk.mode,  # e.g., 'min'
        save_top_k=cfg.checkpoint.topk.k,  # e.g., 5
        save_last=cfg.checkpoint.save_last_ckpt,
        auto_insert_metric_name=False,
        every_n_epochs=cfg.training.checkpoint_every,
    )
    callbacks.append(checkpoint_callback)
    
    # 3. Learning rate monitor
    from pytorch_lightning.callbacks import LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # 4. Progress bar (optional customization)
    from pytorch_lightning.callbacks import TQDMProgressBar
    callbacks.append(TQDMProgressBar(refresh_rate=10))
    
    # Create Trainer
    trainer = pl.Trainer(
        # Training duration
        max_epochs=cfg.training.num_epochs,
        
        # Optimization
        accumulate_grad_batches=cfg.training.gradient_accumulate_every,
        gradient_clip_val=cfg.training.get('gradient_clip_val', None),
        
        # Hardware
        accelerator='gpu',
        devices=1,  # Single GPU for now
        # For multi-GPU: devices=[0, 1], strategy='ddp'
        
        # Precision
        precision='16-mixed' if cfg.training.get('use_fp16', False) else 32,
        
        # Logging
        logger=wandb_logger,
        log_every_n_steps=1,
        
        # Callbacks
        callbacks=callbacks,
        
        # Validation
        check_val_every_n_epoch=cfg.training.val_every,
        val_check_interval=None,  # Only validate at epoch end
        
        # Other
        enable_progress_bar=True,
        enable_model_summary=True,
        default_root_dir=output_dir,
        
        # Debugging (optional)
        # fast_dev_run=True,  # Quick sanity check
        # limit_train_batches=10,  # For testing
        # limit_val_batches=3,
    )
    
    # Train!
    trainer.fit(model, train_dataloader, val_dataloader)
    
    return trainer, model
```

### Phase 5: Update Entry Point (Estimated: 15 min)

Update your hydra-based entry point:

```python
@hydra.main(
    version_base=None,
    config_path='../config',
    config_name='maniflow_pointcloud_policy_robotwin'
)
def main(cfg: OmegaConf):
    # Get output directory
    if 'hydra' in cfg:
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        output_dir = './outputs'
    
    # Train using Lightning
    trainer, model = train_lightning(cfg, output_dir)
    
    print(f"Training complete! Checkpoints saved to: {output_dir}/checkpoints")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {trainer.checkpoint_callback.best_model_score}")

if __name__ == "__main__":
    main()
```

## Testing & Verification

### Step 1: Quick Sanity Check

```python
# In your training script, add fast_dev_run for testing
trainer = pl.Trainer(
    fast_dev_run=5,  # Run 5 train batches, 5 val batches, then exit
    ...
)
```

Run and check:
- [ ] No errors
- [ ] Losses are computed
- [ ] WandB logging works
- [ ] Checkpoint is saved

### Step 2: Short Training Run

```python
trainer = pl.Trainer(
    max_epochs=5,
    limit_train_batches=20,  # Only 20 batches per epoch
    limit_val_batches=5,
    ...
)
```

Compare to baseline:
- [ ] Training loss curve similar shape
- [ ] Validation loss computed correctly
- [ ] Learning rate schedule matches
- [ ] EMA is updating (check callback)

### Step 3: Full Comparison Run

Run full training and compare:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load old WandB logs
old_run = wandb.Api().run("your-entity/maniflow/old_run_id")
old_history = old_run.history()

# Load new WandB logs
new_run = wandb.Api().run("your-entity/maniflow/new_run_id")
new_history = new_run.history()

# Compare training loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(old_history['train_loss'], label='Old Implementation')
plt.plot(new_history['train_loss'], label='Lightning')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Training Loss Comparison')

plt.subplot(1, 2, 2)
plt.plot(old_history['val_loss'], label='Old Implementation')
plt.plot(new_history['val_loss'], label='Lightning')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Validation Loss Comparison')

plt.tight_layout()
plt.savefig('comparison.png')
```

### Checklist for Verification

- [ ] Final validation loss within 1% of baseline
- [ ] Training takes similar wall-clock time (±10%)
- [ ] Memory usage is comparable
- [ ] Checkpoint files load correctly
- [ ] EMA model predictions match (if using EMA)
- [ ] Diffusion sampling metrics are logged
- [ ] Learning rate schedule is identical

## Common Pitfalls & Solutions

### Issue 1: EMA Model Not Updating
**Symptom:** EMA model is None or not improving  
**Solution:** Check that EMACallback is in callbacks list and `use_ema=True`

### Issue 2: Different Number of Training Steps
**Symptom:** Learning rate schedule is off  
**Solution:** Lightning counts steps differently with gradient accumulation.
- Old: step increments every batch
- Lightning: step increments after `accumulate_grad_batches` batches
- Check `trainer.estimated_stepping_batches` matches expected value

### Issue 3: Device Mismatch Errors
**Symptom:** `RuntimeError: Expected all tensors to be on the same device`  
**Solution:** Don't manually call `.to(device)` in training_step - Lightning handles it.
- Exception: In `on_train_epoch_end()` for saved batches, need manual `.to(self.device)`

### Issue 4: Validation Not Running
**Symptom:** No val_loss logged  
**Solution:** Check `check_val_every_n_epoch` and ensure val_dataloader is passed to `trainer.fit()`

### Issue 5: WandB Logging Duplicates
**Symptom:** Metrics logged multiple times  
**Solution:** Check `on_step` and `on_epoch` flags in `self.log()`
- `on_step=True, on_epoch=False` for per-step metrics (train_loss)
- `on_step=False, on_epoch=True` for per-epoch metrics (val_loss)

### Issue 6: Checkpoint Paths Different
**Symptom:** Can't find saved checkpoints  
**Solution:** Lightning saves to `dirpath` in ModelCheckpoint. Update your loading code:
```python
# Old
ckpt_path = os.path.join(output_dir, 'checkpoints', 'topk_0.ckpt')

# New
ckpt_path = trainer.checkpoint_callback.best_model_path
# or
ckpt_path = os.path.join(output_dir, 'checkpoints', 'epoch=0099-val_loss=0.1234.ckpt')
```

## Advanced: Multi-GPU Training

Once single-GPU is working, multi-GPU is trivial:

```python
trainer = pl.Trainer(
    devices=[0, 1, 2, 3],  # Use 4 GPUs
    strategy='ddp',  # Distributed Data Parallel
    ...
)
```

Notes:
- Batch size is per-GPU, so effective batch size = `batch_size * num_gpus`
- Some callbacks may need `rank_zero_only` decorator
- EMA should only update on rank 0

## Resources

- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [LightningModule API](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
- [Trainer API](https://lightning.ai/docs/pytorch/stable/common/trainer.html)
- [Callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html)
- [Logging](https://lightning.ai/docs/pytorch/stable/visualize/logging_intermediate.html)

## Next Steps

1. ✅ Complete baseline training run
2. ✅ Save baseline metrics and checkpoints
3. ⬜ Implement Lightning conversion following this guide
4. ⬜ Run verification tests
5. ⬜ Compare results to baseline
6. ⬜ If successful, deprecate old training code

---

**Good luck! Take your time with the baseline run first.** 🚀

