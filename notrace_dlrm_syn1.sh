source ~/accelsim_work/mgpu/Multigpu-Accelsim/env.sh && source $syn_dlrmpath/env1.sh && export CUDA_VISIBLE_DEVICES=0,1,2,3 && export NCCL_LAUNCH_MODE=GROUP && export FLIP_TO_FULL_TRACING=0 && export PROFILER_API=100 && python3 $syn_dlrmpath/dlrm_s_pytorch.py --data-generation=$DATA_GEN --round-targets=True --learning-rate=1.0 --arch-mlp-bot=$BOT_MLP --arch-mlp-top=$TOP_MLP --arch-sparse-feature-size=$EMB_DIM --max-ind-range=40000000 \
--numpy-rand-seed=727 --num-batches=$NUM_BATCH --data-size 100000000 --num-indices-per-lookup=$EMB_LS --num-indices-per-lookup-fixed=True --arch-embedding-size=$EMB_TBL --print-freq=1 --print-time --mini-batch-size=$BS $EXTRA_FLAGS \
--use-gpu --break-point=10


