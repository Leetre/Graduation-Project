export PYTHONPATH=.
problem=${problem:-score2perf_maestro_absmel2perf_5s_to_30s_aug10x}
#problem=${problem:-score2perf_maestro_language_uncropped_aug}

data_dir=./data/maestro
hparams_set=${hparams_set:-score2perf_transformer_base}
train_dir=./checkpoints/${exp_name}

python tensor2tensor/bin/t2t-datagen \
  --t2t_usr_dir="magenta/models/score2perf" \
  --data_dir=$data_dir \
  --problem=${problem} \
  --alsologtostderr
