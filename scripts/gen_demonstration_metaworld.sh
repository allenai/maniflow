# bash scripts/gen_demonstration_metaworld.sh basketball
# bash scripts/gen_demonstration_metaworld.sh assembly
# bash scripts/gen_demonstration_metaworld.sh bin-picking
# bash scripts/gen_demonstration_metaworld.sh box-close
# bash scripts/gen_demonstration_metaworld.sh button-press-topdown
# bash scripts/gen_demonstration_metaworld.sh button-press-topdown-wall
# bash scripts/gen_demonstration_metaworld.sh button-press
# bash scripts/gen_demonstration_metaworld.sh button-press-wall
# bash scripts/gen_demonstration_metaworld.sh coffee-button
# bash scripts/gen_demonstration_metaworld.sh coffee-pull
# bash scripts/gen_demonstration_metaworld.sh coffee-push
# bash scripts/gen_demonstration_metaworld.sh dial-turn
# bash scripts/gen_demonstration_metaworld.sh disassemble
# bash scripts/gen_demonstration_metaworld.sh door-close
# bash scripts/gen_demonstration_metaworld.sh door-lock
# bash scripts/gen_demonstration_metaworld.sh door-open
# bash scripts/gen_demonstration_metaworld.sh door-unlock
# bash scripts/gen_demonstration_metaworld.sh drawer-close
# bash scripts/gen_demonstration_metaworld.sh drawer-open
# bash scripts/gen_demonstration_metaworld.sh faucet-open
# bash scripts/gen_demonstration_metaworld.sh faucet-close
# bash scripts/gen_demonstration_metaworld.sh hammer
# bash scripts/gen_demonstration_metaworld.sh hand-insert
# bash scripts/gen_demonstration_metaworld.sh handle-press
# bash scripts/gen_demonstration_metaworld.sh handle-pull
# bash scripts/gen_demonstration_metaworld.sh handle-press-side
# bash scripts/gen_demonstration_metaworld.sh handle-pull-side
# bash scripts/gen_demonstration_metaworld.sh lever-pull
# bash scripts/gen_demonstration_metaworld.sh peg-insert-side
# bash scripts/gen_demonstration_metaworld.sh peg-unplug-side
# bash scripts/gen_demonstration_metaworld.sh pick-out-of-hole
# bash scripts/gen_demonstration_metaworld.sh pick-place-wall
# bash scripts/gen_demonstration_metaworld.sh pick-place
# bash scripts/gen_demonstration_metaworld.sh plate-slide-side
# bash scripts/gen_demonstration_metaworld.sh plate-slide-back
# bash scripts/gen_demonstration_metaworld.sh plate-slide-back-side
# bash scripts/gen_demonstration_metaworld.sh plate-slide
# bash scripts/gen_demonstration_metaworld.sh push-back
# bash scripts/gen_demonstration_metaworld.sh push-wall
# bash scripts/gen_demonstration_metaworld.sh push
# bash scripts/gen_demonstration_metaworld.sh reach-wall
# bash scripts/gen_demonstration_metaworld.sh reach
# bash scripts/gen_demonstration_metaworld.sh shelf-place
# bash scripts/gen_demonstration_metaworld.sh soccer
# bash scripts/gen_demonstration_metaworld.sh stick-push
# bash scripts/gen_demonstration_metaworld.sh stick-pull
# bash scripts/gen_demonstration_metaworld.sh sweep
# bash scripts/gen_demonstration_metaworld.sh sweep-into
# bash scripts/gen_demonstration_metaworld.sh window-open
# bash scripts/gen_demonstration_metaworld.sh window-close

# tasks=['assembly', 'bin-picking', 'box-close', 'button-press-topdown', 'button-press-topdown-wall', 'button-press', 'button-press-wall', 'coffee-button', 'coffee-pull', 'coffee-push', 'dial-turn', 'disassemble', 'door-close', 'door-lock', 'door-open', 'door-unlock', 'drawer-close', 'drawer-open', 'faucet-open', 'faucet-close', 'hammer', 'hand-insert', 'handle-press', 'handle-pull', 'handle-press-side', 'handle-pull-side', 'lever-pull', 'peg-insert-side', 'peg-unplug-side', 'pick-out-of-hole', 'pick-place-wall', 'pick-place', 'plate-slide-side', 'plate-slide-back', 'plate-slide-back-side', 'plate-slide', 'push-back', 'push-wall', 'push', 'reach-wall', 'reach', 'shelf-place', 'soccer', 'stick-push', 'stick-pull', 'sweep', 'sweep-into', 'window-open', 'window-close']

cd third_party/Metaworld

task_name=${1}

export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../data/" \

