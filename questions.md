# maniflow discussion
generally how long til convergence?
any sample wandbs?
multi-node or multi-gpu
thoughts on how to RL fine-tune


## arch stuff

**flow vs consistency batch ratio**
- they use 0.75/0.25 everywhere
- does this change for different data amounts? domains?
- trade-off btw flow quality and distillation

**visual_cond_len - this seems important**
- images: 1024 tokens
- point clouds: 128-256 (clean scenes) vs 2048-4096 (cluttered)
- huge memory impact, affects performance
- how do i decide for my setup?

**visual backbone **
- have looked at what it might take to integrate siglip w/projection - seems feasible

**horizon lengths**
- 16 for robotwin, 64 for adroit/dexart
- memory vs temporal modeling tradeoff
- n_action_steps interaction?

**sample_t_mode: beta vs lognorm**
- You said one of them was better in your talk despite the tables - remind me why?

## data & training

**min dataset size**
- 20-50 demos simple tasks
- 100-200 complex
- they got 99.7% w/ 500 demos
- how to know if my task is "simple" or "complex"?

**failure modes to watch**
- val loss curves
- rollout success
- inference speed
- mode collapse / action averaging?
- what's their debug workflow

**temporal ensembling**
- they say critical for real world
- safety + smoothness + execution delays
- what strategy? weighted avg? median?

**augmentations**
- color jitter: brightness 0.3, contrast 0.4, sat 0.5, hue 0.08
- crop 0.95, rotation ±5deg
- prob 0.2 to avoid overfitting
- does this transfer to my domain?

**multi-camera**
- i have wrist_camera_r + head_camera
- sync issues?
- weighting importance?
- active sensing (h1 head movements)

## language stuff

**why is language_conditioned: false everywhere?**
- they have max_lang_cond_len: 1024 configured
- but disabled in robotwin/adroit/dexart
- only metaworld multitask uses it?
- tradeoff: generalization vs performance?
- when is it worth it?

## infrastructure

**gpu memory requirements**
- baseline: n_layer=12, n_emb=768
- large: n_layer=28, n_emb=1152
- batch: 128 (img), 256 (pc)
- how much vram needed? multi-gpu?
- in general 20Gb vram, one 4090 can train. on H100 maybe 200-300 works - might run into other limitations 
- train time
  - for 50 demo: 8-10 hours
  - if large model - 1-2 days (on a single 4090)
  - has observed that maniflow can converge much faster that diffusion models
  - learning rate - best practice is to set this first to 500 (whole train process will be faster)
  - ask follow-up questions about LR once I have some curves
  - 

**sapien version hell**
- dexart needs 2.2.1, robotwin needs 3.0.0b1
- they mention separate envs or manual switching
- actual workflow? having trouble getting it to build (whacking through a dockerfile now)


## my usecase - mjthor spoc

**integration**
- custom h5 + video format
- wrist_camera_r + head_camera
- action: base(3) + head(2) + right_arm(7) + right_gripper(1) = 13dof
- language goals

**specific questions**
- validate my data format assumptions?
- normalization for heterogeneous action spaces (base vs arm vs gripper)
- hierarchical action representations?
- active perception w/ head control?
- long-horizon task composition
- sim (ai2thor/mujoco) to real transfer
