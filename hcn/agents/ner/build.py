import parlai.core.build_data as build_data
import os


def load_nerpa(opt, version=None):
    dpath = opt['ner_directory']
    if not build_data.built(dpath, version_string=version):
        url = os.environ.get('DSTC_NER_URL')
        os.makedirs(dpath, exist_ok=True)
        build_data.download(url, dpath, 'dstc_ner_model.tar.gz')
        build_data.untar(dpath, 'dstc_ner_model.tar.gz')
    build_data.mark_done(dpath, version_string=version)
