# Trio/MultiTrack Music VAE

## Environment

```bash
pip install -e .
```

## Generation binary data

1. Download the LMD-full dataset from https://colinraffel.com/projects/lmd/ and put the MIDI files in `data/lmd/lmd_full`. You can use a subset of LMD for quick start.  

2. 
```bash
# hier-trio_4bar
python magenta/models/music_vae/data_gen.py --input_dir=data/lmd/lmd_full --output_file=data/lmd/trio_4bar --config=hier-trio_4bar

# hier-trio_16bar
python magenta/models/music_vae/data_gen.py --input_dir=data/lmd/lmd_full --output_file=data/lmd/trio_16bar --config=hier-trio_16bar

# hier-multiperf_vel_1bar_med
python magenta/models/music_vae/data_gen.py --input_dir=data/lmd/lmd_full --output_file=data/lmd/multiperf_mel --config=hier-multiperf_vel_1bar_med

# hier-multiperf_vel_1bar_med_chords
python magenta/models/music_vae/data_gen.py --input_dir=data/lmd/lmd_full --output_file=data/lmd/multiperf_mel_chords --config=hier-multiperf_vel_1bar_med_chords
```

## Train model 

```bash
CUDA_VISIBLE_DEVICES=0 exp_name=0923_trio4bar_1 data=trio_4bar model=hier-trio_4bar bash runs/train_mvae.sh

CUDA_VISIBLE_DEVICES=0 exp_name=0923_trio16bar_1 data=trio_16bar model=hier-trio_16bar bash runs/train_mvae.sh

CUDA_VISIBLE_DEVICES=0 exp_name=0923_mel_1 data=multiperf_mel model=hier-multiperf_vel_1bar_med bash runs/train_mvae.sh

CUDA_VISIBLE_DEVICES=0 exp_name=0923_mel_chords_1 data=multiperf_mel_chords model=hier-multiperf_mel_chords bash runs/train_mvae.sh

```

## Generate music
```bash
CUDA_VISIBLE_DEVICES=0 exp_name=0923_trio4bar_1 model=hier-trio_4bar bash runs/test_mvae.sh

```

### Multitrack MusicVAE
Run with the jupyter notebook: `Multitrack_MusicVAE.ipynb`

### MusicVAE
[TODO] `MusicVAE.ipynb`
