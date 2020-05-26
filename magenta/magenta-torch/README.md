# magenta-torch
Pytorch Implementation of MusicVAE with LSTM and GRU architectures and Glow

## run 
Model configuration can be modified in `conf.yml`
1. process data
Download the midi data into `data`, then:
    ```bash
    python scripts/preprocess.py --import_dir=data
    ```
2. train model
    ```bash
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py
    ```
3. generate
Modify the model path to get the best `model.pt`
    ```bash
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --mode=generate
    ```
