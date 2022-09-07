import json
import logging
import random


import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from dataset import Get_HuffPost, Get_Banking, Get_Clinc, Get_Clinc_domain


logger = logging.Logger(__name__)


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, classification_label_id, lm_label_ids, tokens, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.classification_label_id = classification_label_id
        self.lm_label_ids = lm_label_ids
        self.tokens = tokens
        self.label = label
    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def random_word(tokens, tokenizer, select_prob=0.3):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :param select_prob: Probability of selecting for prediction
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []
    tokens = list(tokens)

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < select_prob:
            prob /= select_prob

            # 80% randomly change token to mask token
            if prob < 1.0:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.get_vocab().items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.get_vocab()[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.get_vocab()["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)

    return tokens, output_label


class MetaTask(Dataset):
    def __init__(self, examples, num_task, way, shot, query, tokenizer, training, valuation, args):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples
        # random.shuffle(self.examples)

        self.num_task = num_task
        self.way = way
        self.shot = shot
        self.query = query
        self.tokenizer = tokenizer
        self.max_seq_length = 32
        self.training = training
        self.valuation = valuation
        self.args = args
        self.create_batch(self.num_task, self.training, self.valuation)


    def create_batch(self, num_task, training, valuation):
        self.supports = []  # support set
        self.queries = []  # query set

        if self.args.data_name == 'huffpost':
            train_classes, val_classes, test_classes = Get_HuffPost()
        if self.args.data_name == 'banking':
            train_classes, val_classes, test_classes = Get_Banking()
        if self.args.data_name == 'clinc':
            train_domains, val_domains, test_domains = Get_Clinc_domain()
            train_classes, val_classes, test_classes = Get_Clinc()
        # for each task
        for b in range(num_task):
            if training and not valuation:
                if self.args.data_name == 'clinc':
                    random_domain = np.random.permutation(train_domains)[0]
                    train_examples = [e for e in self.examples if e['domain'] == random_domain]
                    one_train_support = []
                    one_train_query = []
                    assert self.way <= len(train_classes)
                    # random_train_classes = random.sample(train_classes, self.way)
                    random_train_classes = list(numpy.random.permutation(train_classes)[:self.way])
                    ids = numpy.argsort(random_train_classes, kind="stable")
                    for train_class_name in random_train_classes:
                        train_random_examples = [e for e in train_examples if e['label'] == train_class_name]
                        selected_train_examples = numpy.random.permutation(train_random_examples)[
                                                  :self.shot + self.query]
                        train_random_examples_sorted = []
                        for exam in selected_train_examples:
                            exams = {'label': exam['label'],
                                     'context': exam['context'],
                                     'id': ids[random_train_classes.index(train_class_name)]}
                            train_random_examples_sorted.append(exams)

                        random.shuffle(train_random_examples_sorted)
                        exam_train = train_random_examples_sorted[:self.shot]
                        exam_test = train_random_examples_sorted[self.shot:]

                        one_train_support.extend(exam_train)
                        one_train_query.extend(exam_test)

                    self.supports.append(one_train_support)
                    self.queries.append(one_train_query)
                else:
                    one_train_support = []
                    one_train_query = []
                    assert self.way <= len(train_classes)
                    # random_train_classes = random.sample(train_classes, self.way)
                    random_train_classes = list(numpy.random.permutation(train_classes)[:self.way])
                    ids = numpy.argsort(random_train_classes, kind="stable")
                    for train_class_name in random_train_classes:
                        train_random_examples = [e for e in self.examples if e['label'] == train_class_name]
                        selected_train_examples = numpy.random.permutation(train_random_examples)[:self.shot + self.query]
                        train_random_examples_sorted = []
                        for exam in selected_train_examples:
                            exams = {'label': exam['label'],
                                     'context': exam['context'],
                                     'id': ids[random_train_classes.index(train_class_name)]}
                            train_random_examples_sorted.append(exams)

                        random.shuffle(train_random_examples_sorted)
                        # selected_train_examples = random.sample(train_random_examples_sorted, self.shot + self.query)
                        # random.shuffle(selected_train_examples)
                        exam_train = train_random_examples_sorted[:self.shot]
                        exam_test = train_random_examples_sorted[self.shot:]

                        one_train_support.extend(exam_train)
                        one_train_query.extend(exam_test)

                    self.supports.append(one_train_support)
                    self.queries.append(one_train_query)

            if not training and valuation:
                if self.args.data_name == 'clinc':
                    random_domain = np.random.permutation(val_domains)[0]
                    val_examples = [e for e in self.examples if e['domain'] == random_domain]
                    one_val_support = []
                    one_val_query = []
                    assert self.way <= len(val_classes)
                    random_val_classes = list(numpy.random.permutation(val_classes)[:self.way])
                    val_ids = numpy.argsort(random_val_classes, kind="stable")
                    for val_class_name in random_val_classes:
                        val_random_examples = [e for e in val_examples if e['label'] == val_class_name]
                        selected_val_examples = numpy.random.permutation(val_random_examples)[:self.shot + self.query]
                        val_random_examples_sorted = []
                        for val_exam in selected_val_examples:
                            val_exams = {'label': val_exam['label'],
                                         'context': val_exam['context'],
                                         'id': val_ids[random_val_classes.index(val_class_name)]}
                            val_random_examples_sorted.append(val_exams)

                        random.shuffle(val_random_examples_sorted)
                        exam_train = val_random_examples_sorted[:self.shot]
                        exam_test = val_random_examples_sorted[self.shot:]
                        one_val_support.extend(exam_train)
                        one_val_query.extend(exam_test)

                    self.supports.append(one_val_support)
                    self.queries.append(one_val_query)
                else:
                    one_val_support = []
                    one_val_query = []
                    assert self.way <= len(val_classes)
                    random_val_classes = list(numpy.random.permutation(val_classes)[:self.way])
                    val_ids = numpy.argsort(random_val_classes, kind="stable")
                    for val_class_name in random_val_classes:
                        val_random_examples = [e for e in self.examples if e['label'] == val_class_name]
                        selected_val_examples = numpy.random.permutation(val_random_examples)[:self.shot + self.query]
                        val_random_examples_sorted = []
                        for val_exam in selected_val_examples:
                            val_exams = {'label': val_exam['label'],
                                         'context': val_exam['context'],
                                         'id': val_ids[random_val_classes.index(val_class_name)]}
                            val_random_examples_sorted.append(val_exams)

                        random.shuffle(val_random_examples_sorted)
                        # selected_val_examples = random.sample(val_random_examples_sorted, self.shot + self.query)

                        # random.shuffle(selected_val_examples)
                        exam_train = val_random_examples_sorted[:self.shot]
                        exam_test = val_random_examples_sorted[self.shot:]
                        one_val_support.extend(exam_train)
                        one_val_query.extend(exam_test)

                    self.supports.append(one_val_support)
                    self.queries.append(one_val_query)

            if not training and not valuation:
                if self.args.data_name == 'clinc':
                    random_domain = np.random.permutation(test_domains)[0]
                    test_examples = [e for e in self.examples if e['domain'] == random_domain]
                    one_test_support = []
                    one_test_query = []
                    assert self.way <= len(test_classes)
                    random_test_classes = list(numpy.random.permutation(test_classes)[:self.way])
                    test_ids = numpy.argsort(random_test_classes, kind="stable")
                    for test_class_name in random_test_classes:
                        test_random_examples = [e for e in test_examples if e['label'] == test_class_name]
                        selected_test_examples = numpy.random.permutation(test_random_examples)[:self.shot + self.query]
                        test_random_examples_sorted = []

                        for test_exam in selected_test_examples:
                            test_exams = {'label': test_exam['label'],
                                          'context': test_exam['context'],
                                          'id': test_ids[random_test_classes.index(test_class_name)]}
                            test_random_examples_sorted.append(test_exams)
                        random.shuffle(test_random_examples_sorted)
                        exam_train = test_random_examples_sorted[:self.shot]
                        exam_test = test_random_examples_sorted[self.shot:]
                        one_test_support.extend(exam_train)
                        one_test_query.extend(exam_test)

                    self.supports.append(one_test_support)
                    self.queries.append(one_test_query)
                else:
                    one_test_support = []
                    one_test_query = []
                    assert self.way <= len(test_classes)
                    random_test_classes = list(numpy.random.permutation(test_classes)[:self.way])
                    test_ids = numpy.argsort(random_test_classes, kind="stable")
                    for test_class_name in random_test_classes:
                        test_random_examples = [e for e in self.examples if e['label'] == test_class_name]
                        selected_test_examples = numpy.random.permutation(test_random_examples)[:self.shot + self.query]
                        test_random_examples_sorted = []

                        for test_exam in selected_test_examples:
                            test_exams = {'label': test_exam['label'],
                                          'context': test_exam['context'],
                                          'id': test_ids[random_test_classes.index(test_class_name)]}
                            test_random_examples_sorted.append(test_exams)
                        random.shuffle(test_random_examples_sorted)
                        # selected_test_examples = random.sample(test_random_examples_sorted, self.shot + self.query)
                        # random.shuffle(selected_test_examples)

                        exam_train = test_random_examples_sorted[:self.shot]
                        exam_test = test_random_examples_sorted[self.shot:]
                        one_test_support.extend(exam_train)
                        one_test_query.extend(exam_test)

                    self.supports.append(one_test_support)
                    self.queries.append(one_test_query)

    def examples_to_features(self, examples, pad_token=0, mask_padding_with_zero=True):

        features = []

        for (ex_index, example) in enumerate(examples):
            sentence = example['context']
        # inputs: dict
            inputs = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                truncation=True
            )

            # input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
            input_ids = inputs["input_ids"]

            tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in input_ids]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)

            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            classification_label_id = example['id']

            tokens.remove('[CLS]')
            tokens.remove('[SEP]')
            _, t_label = random_word(tokens, self.tokenizer, select_prob=self.args.select_prob)
            lm_label_ids = ([-100] + t_label + [-100])
            lm_label_ids = lm_label_ids + ([-100] * padding_length)

            assert len(input_ids) == self.max_seq_length, \
                "Error with input length {} vs {}".format(len(input_ids), self.max_seq_length)
            assert len(attention_mask) == self.max_seq_length, \
                "Error with input length {} vs {}".format(len(attention_mask), self.max_seq_length)
            assert len(lm_label_ids) == self.max_seq_length, \
                "Error with input length {} vs {}".format(len(input_ids), self.max_seq_length)

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              classification_label_id=classification_label_id,
                              lm_label_ids=lm_label_ids,
                              tokens=tokens,
                              label=example['label']))
        return features

    def create_examples_set(self, examples):

        features = self.examples_to_features(examples=examples)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_classification_label_id = torch.tensor([f.classification_label_id for f in features], dtype=torch.long)
        all_lm_label_ids = torch.tensor([f.lm_label_ids for f in features], dtype=torch.long)
        all_labels = [f.label for f in features]
        all_tokens = [f.tokens for f in features]

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_classification_label_id, all_lm_label_ids)
        return dataset, all_tokens, all_labels

    def __getitem__(self, index):
        support_set, support_tokens, support_set_labels = self.create_examples_set(self.supports[index])
        query_set, query_tokens, query_set_labels = self.create_examples_set(self.queries[index])
        return support_set, query_set, support_tokens, query_tokens, support_set_labels, query_set_labels

    def __len__(self):
        # as we have built up to batch size of sets, you can sample some small batch size of sets.
        return self.num_task
