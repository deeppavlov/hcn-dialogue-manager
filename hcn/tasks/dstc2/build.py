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

os.environ['DATASETS_URL'] = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/'

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

        # Read data and create dictionary of intents
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

        # {'inform_this', 'request_addr', 'confirm_area', 'inform_food',
        # 'deny_food', 'inform_name', 'inform_pricerange', 'request_area',
        # 'deny_name', 'request_pricerange', 'request_food', 'confirm_food',
        # 'confirm_pricerange', 'request_phone', 'request_postcode', 'inform_area'}

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

