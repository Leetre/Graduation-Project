export PYTHONPATH=.
problem=${problem:-score2perf_maestro_language_uncropped_aug}
exp_name=${exp_name:-0705_e3}
DATA_DIR=./data/maestro
hparams_set=${hparams_set:-score2perf_transformer_base}
model=${model:-music_transformer}
TRAIN_DIR=./checkpoints/${exp_name}

gpu=${gpu:-4}
if [[ $CUDA_VISIBLE_DEVICES != "" ]]; then
  t=(${CUDA_VISIBLE_DEVICES//,/ })
  gpu=${#t[@]}
fi

echo "Using #gpu=$gpu..."

HPARAMS=\
"label_smoothing=0.0,"\
"max_length=0,"\
"max_target_seq_length=512"

#HPARAMS=$HPARAMS",self_attention_type=dot_product_relative_v2,"\
#"max_relative_position=1024"

HPARAMS=$HPARAMS",sampling_method=random"

DECODE_HPARAMS=\
"alpha=0,"\
"beam_size=1,"\
"extra_length=512"

t2t_decoder \
  --data_dir="${DATA_DIR}" \
  --decode_hparams="${DECODE_HPARAMS}" \
  --hparams="$HPARAMS" \
  --hparams_set=${hparams_set} \
  --model=${model} \
  --problem=${problem} \
  --output_dir=${TRAIN_DIR} 2>&1 | tee -a $TRAIN_DIR/test_log.txt
#  --decode_interactive \
