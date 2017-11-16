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
        filename = 'dstc2.tar.gz'

        # Download the data.
        print('Trying to download a dataset %s from the repository' % filename)
        url = urllib.parse.urljoin(ds_path, filename)
        if url.startswith('file://'):
            build_data.move(url[7:], dpath)
        else:
            build_data.download(url, dpath, filename)
        build_data.untar(dpath, filename)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)

