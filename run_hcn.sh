#!/usr/bin/env bash

# initialize dataset paths
export DATASETS_URL=http://share.ipavlov.mipt.ru:8080/repository/datasets/
export DSTC_NER_URL=http://lnsigo.mipt.ru/export/ner_dstc_model.tar.gz

# display data example
python3 ./utils/display_data.py -t dialog_babi:task:6\
                                -n 10
python3 ./utils/display_data.py -t hcn.tasks.dstc2.agents\
                                -dt train:ordered\
                                -n 10

# make directories
mkdir -p ./build ./data

# download glove embeddings (to use bot without embeddings ommit `embedding-file`
# parameter)
if [ ! -f ./data/glove.42B.300d.txt ]; then
    wget -O ./data/glove.42B.300d.zip "http://nlp.stanford.edu/data/glove.42B.300d.zip"
    unzip -d ./data ./data/glove.42B.300d.zip && rm ./data/glove.42B.300d.zip
fi

# train while valid score improves
python3 ./utils/train_model.py -t hcn.tasks.dstc2.agents\
                               -mf ./build/hcn\
                               -dt train:ordered\
                               -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               --slot-model hcn.agents.ner.nerpa:NerProcessingAgent\
                               --template-file dstc2/dstc2-templates.txt\
                               --template-path-relative true\
                               --embedding-file ./data/glove.42B.300d.txt\
                               --num-epochs -1\
                               --log-every-n-secs -1\
                               --log-every-n-epochs 1\
                               --learning-rate .02\
                               --hidden-dim 128\
                               --validation-every-n-epochs 1\
                               --chosen-metric accuracy\
                               -dbf true\
                               -vp 10

# interactive evaluate
python3 ./utils/interactive.py -m hcn.agents.hcn.hcn:HybridCodeNetworkAgent\
                               --pretrained-model ./build/hcn\
                               --slot-model hcn.agents.ner.nerpa:NerProcessingAgent\
                               --template-file dstc2/dstc2-templates.txt\
                               --template-path-relative true\
                               --embedding-file ./data/glove.42B.300d.txt\
