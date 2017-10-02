#!/usr/bin/env bash

# display data example
python3 ./utils/display_data.py -t dialog_babi:task:6 -n 10

# build directory
mkdir -p ./build

# train for 24 epochs
python3 ./utils/train_model.py -t dialog_babi:task:6\
                               -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               -mf ./build/hcn\
                               --datatype train:ordered\
                               --num-epochs -1\
                               --log-every-n-secs -1\
                               --log-every-n-epochs 1\
                               --learning-rate .005\
                               --hidden-dim 128\
                               --validation-every-n-epochs 1\
                               -dbf true\
                               --chosen-metric accuracy\
                               --tracker babi6\
                               --action-mask true\
                               -vp 10

# interactive evaluate
python3 ./utils/interactive.py -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               --pretrained-model ./build/hcn\
                               --dict-file ./build/hcn.dict\
                               --tracker babi6\
                               --action-mask true

