import json
import os
import pkg_resources
from fuzzywuzzy import process
from utils.ner_src.corpus import Corpus
from utils.ner_src.network import NER


def get_modelfile(fname):
    rel_path = os.path.join('model', fname)
    return pkg_resources.resource_filename(__name__, rel_path)


class Nerpa:
    def __init__(self,
                 dict_filepath=get_modelfile('dict.txt'),
                 model_filepath=get_modelfile('ner_model.ckpt'),
                 params_filepath=get_modelfile('params.json'),
                 slot_vals_filepath=get_modelfile('slot_vals.json')):
        # Build and initialize the model
        with open(params_filepath) as f:
            network_params = json.load(f)
        self._corpus = Corpus(dicts_filepath=dict_filepath)
        self._ner_network = NER(self._corpus, pretrained_model_filepath=model_filepath, **network_params)
        with open(slot_vals_filepath) as f:
            self._slot_vals = json.load(f)

    def predict_slots(self, utterance):
        # For utterance extract named entities and perform normalization for slot filling
        tokens = utterance.split()
        tags = self._ner_network.predict_for_token_batch([tokens])[0]
        entities, slots = self._chunk_finder(tokens, tags)
        slot_values = dict()
        for entity, slot in zip(entities, slots):
            slot_values[slot] = self.ner2slot(entity, slot)
        return slot_values

    def ner2slot(self, input_entity, slot):
        # Given named entity return normalized slot value
        if isinstance(input_entity, list):
            input_entity = ' '.join(input_entity)
        entities = list()
        normalized_slot_vals = list()
        for entity_name in self._slot_vals[slot]:
            for entity in self._slot_vals[slot][entity_name]:
                entities.append(entity)
                normalized_slot_vals.append(entity_name)
        best_match = process.extract(input_entity, entities, limit=2 ** 20)[0][0]
        return normalized_slot_vals[entities.index(best_match)]

    @staticmethod
    def _chunk_finder(tokens, tags):
        # For BIO labeled sequence of tags extract all named entities form tokens
        # Example
        prev_tag = ''
        chunk_tokens = list()
        entities = list()
        slots = list()
        for token, tag in zip(tokens, tags):
            curent_tag = tag.split('-')[-1]
            current_prefix = tag.split('-')[0]
            if tag.startswith('B-'):
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = list()
                chunk_tokens.append(token)
            if current_prefix == 'I':
                if curent_tag != prev_tag:
                    if len(chunk_tokens) > 0:
                        entities.append(' '.join(chunk_tokens))
                        slots.append(prev_tag)
                        chunk_tokens = list()
                else:
                    chunk_tokens.append(token)
            if current_prefix == 'O':
                if len(chunk_tokens) > 0:
                    entities.append(' '.join(chunk_tokens))
                    slots.append(prev_tag)
                    chunk_tokens = list()
            prev_tag = curent_tag

        return entities, slots
