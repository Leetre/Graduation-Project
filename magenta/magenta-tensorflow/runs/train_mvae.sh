exp_name=${exp_name:-"test_mvae"}
model=${model:-"hier-multiperf_vel_1bar_med"}
# hier-multiperf_vel_1bar_med_chords
hparams=${hparams:-}
data=${data:-"lmd_full5"}

num_gpu=1
if [[ $CUDA_VISIBLE_DEVICES != "" ]]; then
  t=(${CUDA_VISIBLE_DEVICES//,/ })
  num_gpu=${#t[@]}
fi
echo $num_gpu

python magenta/models/music_vae/music_vae_train.py \
  --config=$model \
  --run_dir=checkpoints/$exp_name \
  --mode=train \
  --num_sync_workers=$num_gpu \
  --examples_path=data/lmd/${data}_*.tfrecord* \
  --hparams="$hparams"
