#export LD_PRELOAD=$CONDA_PREFIX/lib/libiomp5.so:$CONDA_PREFIX/lib/libjemalloc.so0
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
#export KMP_AFFINITY=verbose,granularity=fine,compact,1,0
#export KMP_BLOCKTIME=1
#export OMP_NUM_THREADS=1

PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'
PyGetCore='import sys; c=int(sys.argv[1]); print(",".join(str(2*i) for i in range(c)))'
PyGetHT='import sys; c=int(sys.argv[1]); print(",".join(str(2*i + off) for off in (0, 48) for i in range(c)))'
NUM_BATCH=120
BS=64
LOG=print_out.log
INSTANCES=1

DLRM_SYSTEMS=$PWD
MODELS_PATH=$PWD

#BOT_MLP=256-128-128
#TOP_MLP=128-64-1
#EMBS='128,1000000,60,120'

BOT_MLP=2048-1024-256-128
TOP_MLP=512-256-1
EMBS='128,1000000,170,180'
#EMBS='128'

DLRM_SYSTEMS='/home/tgrogers-raid/a/chris241/accelsim_work/mgpu/DLRM_synthetic'
echo $MODELS_PATH
#echo $PyGenTbl
for e in $EMBS; do
    IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
    echo $EMB_DIM
    echo $EMB_ROW
    EMB_TBL=$(python -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
    DATA_GEN="prod,$DLRM_SYSTEMS/datasets/reuse_low/table_1M.txt,$EMB_ROW"
    #C=$(python -c "$PyGetCore" "$INSTANCES")
    

    python3 $MODELS_PATH/dlrm_s_pytorch.py --data-generation=$DATA_GEN --round-targets=True --learning-rate=1.0 --arch-mlp-bot=$BOT_MLP --arch-mlp-top=$TOP_MLP --arch-sparse-feature-size=$EMB_DIM --max-ind-range=40000000 --numpy-rand-seed=727 --num-batches=$NUM_BATCH --data-size 100000000 --num-indices-per-lookup=$EMB_LS --num-indices-per-lookup-fixed=True --arch-embedding-size=$EMB_TBL --print-freq=10 --print-time --mini-batch-size=$BS $EXTRA_FLAGS --use-gpu | tee -a $LOG
done
