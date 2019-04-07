GPU_ID=0
NET="res101"
TRAIN_IMDB="voc_2007_trainval"
TEST_IMDB="voc_2007_test"
ITERS=70000
ANCHORS="[8,16,32]"
RATIOS="[0.5,1,2]"
STEPSIZE="[50000]"
EXTRA_ARGS=${array[@]:3:$len}
CUDA_VISIBLE_DEVICES=${GPU_ID}
time python ./tools/trainval_net.py \
  --weight data/imagenet_weights/${NET}.ckpt \
  --imdb ${TRAIN_IMDB} \
  --imdbval ${TEST_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${NET}.yml \
  --net ${NET} \
  --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
  TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}