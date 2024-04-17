PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'
PyGetCore='import sys; c=int(sys.argv[1]); print(",".join(str(2*i) for i in range(c)))'
PyGetHT='import sys; c=int(sys.argv[1]); print(",".join(str(2*i + off) for off in (0, 48) for i in range(c)))'

DLRM_SYSTEMS=$PWD
MODELS_PATH=$PWD

#### Hyper Parameters that might need to change
REUSE_LEVEL='low' #'medium', 'high'
NUM_BATCH=120
BS=6144 #8192
LOG=print_out.log
INSTANCES=1
DLRM_SYSTEMS='/home/tgrogers-raid/a/liu2550/DLRM_synthetic'

#### Model Parameters
BOT_MLP=1024-512-128-128
TOP_MLP=384-192-1
EMBS='128,1000000,120,150'

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_LAUNCH_MODE=GROUP
export FLIP_TO_FULL_TRACING=0 
export PROFILER_API=100

for e in $EMBS; do
    IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
    EMB_TBL=$(python -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
    DATA_GEN="prod,$DLRM_SYSTEMS/datasets/reuse_$REUSE_LEVEL/table_1M.txt,$EMB_ROW"

    python3.7 dlrm_s_pytorch.py --data-generation=$DATA_GEN --round-targets=True --learning-rate=1.0 --arch-mlp-bot=$BOT_MLP --arch-mlp-top=$TOP_MLP --arch-sparse-feature-size=$EMB_DIM --max-ind-range=40000000 \
            --numpy-rand-seed=727 --num-batches=$NUM_BATCH --data-size 100000000 --num-indices-per-lookup=$EMB_LS --num-indices-per-lookup-fixed=True --arch-embedding-size=$EMB_TBL --print-freq=1 --print-time --mini-batch-size=$BS $EXTRA_FLAGS \
            --use-gpu --break-point=10
done