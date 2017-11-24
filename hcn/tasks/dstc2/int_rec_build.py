"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import parlai.core.build_data as build_data
import os
import urllib
import pickle
import json
import os
import numpy as np
import re

os.environ['DATASETS_URL'] = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/'

def data_preprocessing(f):
    """Preprocess the data.

    Args:
        f: list of text samples

    Returns:
        preprocessed list of text samples
    """
    f = [x.lower() for x in f]
    f = [x.replace("\\n", " ") for x in f]
    f = [x.replace("\\t", " ") for x in f]
    f = [x.replace("\\xa0", " ") for x in f]
    f = [x.replace("\\xc2", " ") for x in f]

    f = [re.sub('!!+', ' !! ', x) for x in f]
    f = [re.sub('!', ' ! ', x) for x in f]
    f = [re.sub('! !', '!!', x) for x in f]

    f = [re.sub('\?\?+', ' ?? ', x) for x in f]
    f = [re.sub('\?', ' ? ', x) for x in f]
    f = [re.sub('\? \?', '??', x) for x in f]

    f = [re.sub('\?!+', ' ?! ', x) for x in f]

    f = [re.sub('\.\.+', '..', x) for x in f]
    f = [re.sub('\.', ' . ', x) for x in f]
    f = [re.sub('\.  \.', '..', x) for x in f]

    f = [re.sub(',', ' , ', x) for x in f]
    f = [re.sub(':', ' : ', x) for x in f]
    f = [re.sub(';', ' ; ', x) for x in f]
    f = [re.sub('\%', ' % ', x) for x in f]

    f = [x.replace("$", "s") for x in f]
    f = [x.replace(" u ", " you ") for x in f]
    f = [x.replace(" em ", " them ") for x in f]
    f = [x.replace(" da ", " the ") for x in f]
    f = [x.replace(" yo ", " you ") for x in f]
    f = [x.replace(" ur ", " your ") for x in f]
    f = [x.replace("you\'re", "you are") for x in f]
    f = [x.replace(" u r ", " you are ") for x in f]
    f = [x.replace("yo\'re", " you are ") for x in f]
    f = [x.replace("yu\'re", " you are ") for x in f]
    f = [x.replace("u\'re", " you are ") for x in f]
    f = [x.replace(" urs ", " yours ") for x in f]
    f = [x.replace("y'all", "you all") for x in f]

    f = [x.replace(" r u ", " are you ") for x in f]
    f = [x.replace(" r you", " are you") for x in f]
    f = [x.replace(" are u ", " are you ") for x in f]

    f = [x.replace("won't", "will not") for x in f]
    f = [x.replace("can't", "cannot") for x in f]
    f = [x.replace("i'm", "i am") for x in f]
    f = [x.replace(" im ", " i am ") for x in f]
    f = [x.replace("ain't", "is not") for x in f]
    f = [x.replace("'ll", " will") for x in f]
    f = [x.replace("'t", " not") for x in f]
    f = [x.replace("'ve", " have") for x in f]
    f = [x.replace("'s", " is") for x in f]
    f = [x.replace("'re", " are") for x in f]
    f = [x.replace("'d", " would") for x in f]

    # stemming
    f = [re.sub("ies( |$)", "y ", x) for x in f]
    f = [re.sub("s( |$)", " ", x) for x in f]
    f = [re.sub("ing( |$)", " ", x) for x in f]
    f = [x.replace("tard ", " ") for x in f]

    f = [re.sub(" [*$%&#@][*$%&#@]+", " xexp ", x) for x in f]
    f = [re.sub(" [0-9]+ ", " DD ", x) for x in f]
    f = [re.sub("<\S*>", "", x) for x in f]
    f = [re.sub('\s+', ' ', x) for x in f]
    return f


def write_input_fasttext_emb(data, path, data_name):
    """Write down input files for fasttext embedding.

    Args:
        data: array of text samples
        path: path to folder to put the files
        data_name: mode of writing files "train" or "test"

    Returns:
        nothing
    """
    f = open(path + '_fasttext_emb.txt', 'w')
    for i in range(len(data)):
        if data_name == 'train' or data_name == 'test':
            f.write(data[i] + '\n')
        else:
            print('[ Incorrect data name for writing fasttext embedding input]')
    f.close()


def build(opt):
    dpath = os.path.join(opt['datapath'], 'dstc2')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove there outdates files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        ds_path = os.environ.get('DATASETS_URL')
        print('URL:', ds_path)
        filename = 'dstc2.tar.gz'

        # Download the data.
        print('Trying to download a dataset %s from the repository' % filename)
        url = urllib.parse.urljoin(ds_path, filename)
        if url.startswith('file://'):
            build_data.move(url[7:], dpath)
        else:
            build_data.download(url, dpath, filename)
        build_data.untar(dpath, filename, deleteTar=False)

        # Create dictionary of intents
        print("[loading dstc2-dialog data:" + dpath + " to create dictionary of intents]")
        all_intents = []
        with open(os.path.join(dpath, 'dstc2-trn.jsonlist')) as read:
            for line in read:
                line = line.strip()
                # if empty line - it is the end of dialog
                if not line:
                    continue

                replica = json.loads(line)
                if 'goals' not in replica.keys():
                    # bot reply
                    continue
                if replica['dialog-acts']:
                    for act in replica['dialog-acts']:
                        for slot in act['slots']:
                            if slot[0] == 'slot':
                                all_intents.append(act['act'] + '_' + slot[1])
                            else:
                                all_intents.append(act['act'] + '_' + slot[0])
                        if len(act['slots']) == 0:
                            all_intents.append(act['act'])


        with open(os.path.join(dpath, 'dstc2-val.jsonlist')) as read:
            for line in read:
                line = line.strip()
                # if empty line - it is the end of dialog
                if not line:
                    continue

                replica = json.loads(line)
                if 'goals' not in replica.keys():
                    # bot reply
                    continue
                if replica['dialog-acts']:
                    for act in replica['dialog-acts']:
                        for slot in act['slots']:
                            if slot[0] == 'slot':
                                all_intents.append(act['act'] + '_' + slot[1])
                            else:
                                all_intents.append(act['act'] + '_' + slot[0])
                        if len(act['slots']) == 0:
                            all_intents.append(act['act'])

        intents = set(all_intents)
        with open(os.path.join(dpath, "intents.txt"), "wb") as fp:  # Pickling
            pickle.dump(intents, fp)

        # Prepare data for fasttext
        train_data = []
        intents = np.array(list(intents))
        print(intents)

        with open(os.path.join(dpath, 'dstc2-trn.jsonlist')) as read:
            for line in read:
                line = line.strip()
                # if empty line - it is the end of dialog
                if not line:
                    continue

                replica = json.loads(line)
                if replica['dialog-acts']:
                    train_data.append(replica['text'])


        with open(os.path.join(dpath, 'dstc2-val.jsonlist')) as read:
            for line in read:
                line = line.strip()
                # if empty line - it is the end of dialog
                if not line:
                    continue

                replica = json.loads(line)
                if replica['dialog-acts']:
                    train_data.append(replica['text'])

        train_data = data_preprocessing(train_data)
        write_input_fasttext_emb(train_data,
                                 path=os.path.join(dpath, 'intents_dstc2_train'),
                                 data_name='train')

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

