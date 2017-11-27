#!/usr/bin/env bash

# display data example
python3 ./utils/display_data.py -t dialog_babi:task:6\
                                -n 10
DATASETS_URL=http://share.ipavlov.mipt.ru:8080/repository/datasets/\
    python3 ./utils/display_data.py  -t hcn.tasks.dstc2.agents\
                                     -dt train:ordered\
                                     -n 10

# build directory
mkdir -p ./build

# train for 24 epochs
python3 ./utils/train_model.py -t hcn.tasks.dstc2.agents\
                               -mf ./build/hcn\
                               -dt train:ordered\
                               -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               --slot-model hcn.agents.ner.nerpa:NerProcessingAgent\
                               --template-file ./data/dstc2-templates.txt\
                               --embedding-file ./data/glove.42B.300d.txt\
                               --num-epochs -1\
                               --log-every-n-secs -1\
                               --log-every-n-epochs 1\
                               --learning-rate .05\
                               --hidden-dim 128\
                               --validation-every-n-epochs 1\
                               --chosen-metric accuracy\
                               -dbf true\
                               -vp 10

# interactive evaluate
python3 ./utils/interactive.py -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               --pretrained-model ./build/hcn\
                               --slot-model hcn.agents.ner.nerpa:NerProcessingAgent\
                               --template-file ./data/dstc2-templates.txt\
                               --embedding-file ./data/glove.42B.300d.txt\
