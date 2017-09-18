#!/usr/bin/env bash

# display data example
python3 ./utils/display_data.py -t dialog_babi:task:6 -n 10

# build directory
mkdir -p ./build

# train for 24 epochs
python3 ./utils/train_model.py -t dialog_babi:task:5\
                               -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               -mf ./build/hcn\
                               --datatype train:ordered\
                               --num-epochs 24\
                               --log-every-n-secs -1\
                               --log-every-n-epochs -1\
                               --learning-rate .1\
                               --hidden-dim 128\
                               --validation-every-n-epochs 5

# interactive evaluate
python3 ./utils/interactive.py -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               --pretrained-model ./build/hcn-91700\
                               --dict-file ./build/hcn.dict

