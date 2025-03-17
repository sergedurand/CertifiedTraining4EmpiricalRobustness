#!/bin/bash

python train.py --method=fast --dir=model_cifar --scheduler_opts=start=0,length=0 --num-epochs=30 --model=preactresnet18 --test_att_n_steps 7 --test_att_step_size 0.25 --test-interval 3 --datadir data --eps 0.047058823529411764 --batch-size 128 --device cuda --seed 0 --scheduler_name SmoothedScheduler --attack orig-nfgsm --lr-step batch --grad-norm inf --weight-decay 5e-4 --lr-schedule cyclic --opt SGD --coef_scheduler_opts=start=0,length=0 --data CIFAR --unif 2  --lr-max 0.2 --only_adv --init-method pass --l1_coeff 0 --reg-lambda 0.0
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=0,length=0 --num-epochs=30 --model=preactresnet18 --test_att_n_steps 7 --test_att_step_size 0.25 --test-interval 3 --datadir data --eps 0.06274509803921569 --batch-size 128 --device cuda --seed 0 --scheduler_name SmoothedScheduler --attack orig-nfgsm --lr-step batch --grad-norm inf --weight-decay 5e-4 --lr-schedule cyclic --opt SGD --coef_scheduler_opts=start=0,length=0 --data CIFAR --unif 2  --lr-max 0.2 --only_adv --init-method pass --l1_coeff 0 --reg-lambda 0.0
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=0,length=0 --num-epochs=30 --model=preactresnet18 --test_att_n_steps 7 --test_att_step_size 0.25 --test-interval 3 --datadir data --eps 0.0784313725490196 --batch-size 128 --device cuda --seed 0 --scheduler_name SmoothedScheduler --attack orig-nfgsm --lr-step batch --grad-norm inf --weight-decay 5e-4 --lr-schedule cyclic --opt SGD --coef_scheduler_opts=start=0,length=0 --data CIFAR --unif 2  --lr-max 0.2 --only_adv --init-method pass --l1_coeff 0 --reg-lambda 0.0
python train.py --method=fast --dir=model_cifar --scheduler_opts=start=0,length=0 --num-epochs=30 --model=preactresnet18 --test_att_n_steps 7 --test_att_step_size 0.25 --test-interval 3 --datadir data --eps 0.09411764705882353 --batch-size 128 --device cuda --seed 0 --scheduler_name SmoothedScheduler --attack orig-nfgsm --lr-step batch --grad-norm inf --weight-decay 5e-4 --lr-schedule cyclic --opt SGD --coef_scheduler_opts=start=0,length=0 --data CIFAR --unif 2  --lr-max 0.2 --only_adv --init-method pass --l1_coeff 0 --reg-lambda 0.0
