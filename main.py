import argparse
import datetime
import gc
import glob
import json
import logging
from copy import deepcopy

import numpy
import os
import random
import time
from collections import OrderedDict

from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from MLM import MaskedLMTransformerClassification
from dataset import MetaSet_Split, Read_HuffPost, Get_HuffPost, Read_Banking, Get_Banking, Get_Clinc, Read_Clinc
from log import Logger
from metatask import MetaTask
from torch.nn import functional as F

with torch.cuda.device(1):
    pass


def create_batch_of_tasks(tasksets, batch_size, is_shuffle=False):
    idxs = list(range(0, len(tasksets)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        yield [tasksets[idxs[i]] for i in range(i, min(i + batch_size, len(tasksets)))]


def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    numpy.random.seed(value)
    random.seed(value)


def print_args(args):
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{} = {}".format(attr.upper(), value))
    print("""
                       .::::.
                     .::::::::.
                    :::::::::::
                 ..:::::::::::'
              '::::::::::::'
                .::::::::::
           '::::::::::::::..
                ..::::::::::::.
              ``::::::::::::::::
               ::::``:::::::::'        .:::.
              ::::'   ':::::'       .::::::::.
            .::::'      ::::     .:::::::'::::.
           .:::'       :::::  .:::::::::' ':::::.
          .::'        :::::.:::::::::'      ':::::.
         .::'         ::::::::::::::'         ``::::.
     ...:::           ::::::::::::'              ``::.
    ```` ':.          ':::::::::'                  ::::..
                       '.:::::'                    ':'````..
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_name", default='banking', type=str,
                        help="[banking, huffpost, clinc]")

    parser.add_argument("--bert_pretrain_path", default='bert_base', type=str,
                        help="Path to bert model")

    parser.add_argument("--output_dir", default='./result', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--training_batch_size", default=4, type=int,
                        help="Total batch size for training.")

    parser.add_argument("--test_batch_size", default=20, type=int,
                        help="Total batch size for test.")

    parser.add_argument("--way", type=int, default=10,
                        help="Classes for each task")

    parser.add_argument("--shot", type=int, default=5,
                        help="Support examples for each class for each task")

    parser.add_argument("--query", type=int, default=25,
                        help="Query examples for each class for each task")

    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_update_step", default=5, type=int,
                        help="the step of inner gradient updating")

    parser.add_argument("--random_seed_value", type=int, default=2022,
                        help="specify random seed value")

    parser.add_argument("--train_epochs", default=15, type=int,
                        help="Max number of training epochs")

    parser.add_argument("--num_task_train", default=100, type=int,
                        help="Total number of meta tasks for training epoch")

    parser.add_argument("--num_task_val", default=20, type=int,
                        help="Total number of meta tasks for validation")

    parser.add_argument("--num_task_test", default=1000, type=int,
                        help="Total number of tasks for testing")

    parser.add_argument("--do_train", default=True,
                        help="Whether to train")

    parser.add_argument("--do_valuation", default=True,
                        help="Whether to valuate")

    parser.add_argument("--do_test", default=True,
                        help="Whether to test")

    parser.add_argument("--store_results", default=True,
                        help="Whether to store results")

    parser.add_argument("--use_saving_model", default=True,
                        help="Whether to use already saving model")

    parser.add_argument("--fewrel", default=False,
                        help="Whether to use fewrel dataset")

    parser.add_argument("--select_prob", default=0.3, type=float,
                        help="The prob of random words")

    parser.add_argument("--test_lm_loss", default=True,
                        help="Whether to add the lm loss to the end loss")

    parser.add_argument("--rho", default=0.001, type=float,
                        help="The trade-off of auxiliary task")

    parser.add_argument("--optim", default='Adamw', type=str,
                        help="Select the optimizer for training in [Adamw, Adam]")

    args = parser.parse_args()
    print_args(args)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dir_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    log_dir = './logs/' + args.data_name + '_' + str(args.way) + '_way_' + str(args.shot) + '_shot_' + \
              'train_epochs_' + str(args.train_epochs) + '_random_word_prob_' + str(args.select_prob) + \
              '_rho_' + str(args.rho)
    file = 'Train_' + str(args.do_train) + '_test_lm_loss_' + str(args.test_lm_loss) + '_' + dir_time + 'que_sup.log'

    logger = Logger(log_dir, file, True)
    logger.append(vars(args))

    random_seed(args.random_seed_value)
    if args.data_name == 'huffpost':
        save_path = '/huffpost_{}-way {}-shot-test'.format(args.way, args.shot)
    if args.data_name == 'banking':
        save_path = '/banking_data_{}-way {}-shot-test'.format(args.way, args.shot)
    if args.data_name == 'clinc':
        save_path = '/clinc_{}-way {}-shot-test'.format(args.way, args.shot)

    output_dir = args.output_dir + '/_' + args.data_name + '_' + str(args.way) + '_way_' + str(args.shot) + '_shot_' \
                 + 'train_epochs_' + str(args.train_epochs) + '_random_word_prob_' + str(args.select_prob) \
                 + '_rho_' + str(args.rho)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.data_name == 'huffpost':
        all_data = Read_HuffPost('./dataset/HuffPost/huffpost.json')
    if args.data_name == 'banking':
        all_data = Read_Banking('./dataset/banking_data/categories.json',
                                './dataset/banking_data/train.csv',
                                './dataset/banking_data/test.csv')
    if args.data_name == 'clinc':
        all_data = Read_Clinc('./dataset/clinc150.json')
    if args.data_name == 'huffpost':
        train_classes, val_classes, test_classes = Get_HuffPost()
    if args.data_name == 'banking':
        train_classes, val_classes, test_classes = Get_Banking()
    if args.data_name == 'clinc':
        train_classes, val_classes, test_classes = Get_Clinc()

    train_data, val_data, test_data = MetaSet_Split(all_data, train_classes, val_classes, test_classes, args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path, do_lower_case=True)

    model = MaskedLMTransformerClassification(args, device)

    total_step = (args.num_task_train / args.training_batch_size) * 30

    if args.optim == 'Adamw':
        out_opti = torch.optim.AdamW(model.parameters(),
                                     lr=args.lr,
                                     betas=(0.9, 0.98),
                                     eps=1e-6,
                                     weight_decay=0.1)
    if args.optim == 'Adam':
        out_opti = torch.optim.Adam(model.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(out_opti, num_warmup_steps=0, num_training_steps=total_step)

    if args.do_train:
        print("-"*20, "Training Starting", "-"*20)
        count = 1
        best_acc = 0
        cycle = 0
        all_loss_list = []
        task_acc_list = []
        for epoch in range(args.train_epochs):
            train = MetaTask(train_data,
                             num_task=args.num_task_train,
                             way=args.way,
                             shot=args.shot,
                             query=args.query,
                             tokenizer=tokenizer,
                             training=True,
                             valuation=False,
                             args=args)

            batch_of_train_tasks = create_batch_of_tasks(train,
                                                         batch_size=args.training_batch_size,
                                                         is_shuffle=True)
            grad_for_cm_train = []
            for step, task_batch in enumerate(batch_of_train_tasks):
                task_acc = []
                gradients = []
                lm_gradients = []
                q_gradients = []

                for task_id, task in enumerate(task_batch):
                    support = task[0]
                    query = task[1]
                    support_dataloader = DataLoader(support, batch_size=len(support))

                    train_model = deepcopy(model)

                    if args.optim == 'Adamw':
                        inner_optimizer = torch.optim.AdamW(train_model.parameters(),
                                                            lr=args.lr,
                                                            betas=(0.9, 0.98),
                                                            eps=1e-6,
                                                            weight_decay=0.1)
                    if args.optim == 'Adam':
                        inner_optimizer = torch.optim.Adam(train_model.parameters(), lr=args.lr)

                    train_model.to(device)
                    train_model.train()

                    for i in range(args.num_update_step):
                        all_loss = []
                        for sup_step, batch in enumerate(support_dataloader):
                            inner_optimizer.zero_grad()
                            batch = tuple(t.to(device) for t in batch)
                            input_ids, attention_mask, classification_label_id, lm_label_ids = batch
                            cm_loss, lm_loss, _ = train_model(input_ids=input_ids,
                                                              attention_mask=attention_mask,
                                                              classification_label_id=classification_label_id,
                                                              lm_label_ids=lm_label_ids,
                                                              pos1=None,
                                                              pos2=None)

                            train_loss = cm_loss * (1 - args.rho) + lm_loss * args.rho
                            train_loss.backward()
                            all_loss.append(train_loss.item())

                            logger.append("epoch:{}, step:{}, task_id:{}, number of updating:{}, Base Learner Loss:{:.6f}".format(
                                    epoch, step, task_id, i, numpy.mean(all_loss)))

                            if i == 4:
                                all_loss_list.append(numpy.mean(all_loss))
                            inner_optimizer.step()

                    weight_before_ = list(model.cm.parameters())
                    weight_after_ = list(train_model.cm.parameters())
                    for i, (params_before, params_after) in enumerate(zip(weight_before_, weight_after_)):
                        gradient = params_before - params_after
                        if task_id == 0:
                            gradients.append(gradient)
                        else:
                            gradients[i] += gradient

                    query_dataloader = DataLoader(query, batch_size=len(query), shuffle=False)
                    query_batch = iter(query_dataloader).next()
                    query_batch = tuple(t.to(device) for t in query_batch)

                    input_ids, attention_mask, classification_label_id, lm_label_ids = query_batch
                    cm_output = train_model.cm(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               labels=classification_label_id)
                    cm_logits = F.softmax(cm_output[1], dim=1)

                    train_model.eval()
                    with torch.no_grad():
                        p_label_ids = torch.argmax(cm_logits, dim=1)
                        p_label_ids = p_label_ids.detach().cpu().numpy().tolist()
                        labels_ids = classification_label_id.detach().cpu().numpy().tolist()
                        accuracy = accuracy_score(labels_ids, p_label_ids)
                        task_acc.append(accuracy)
                    task_acc_list.append(numpy.mean(task_acc))

                    q_gradient = torch.autograd.grad(cm_output[0], train_model.cm.parameters(), allow_unused=True)
                    if task_id == 0:
                        q_gradients = list(q_gradient)
                    else:
                        for i, q in enumerate(q_gradients):
                            q_gradients[i] += list(q_gradient)[i]

                    train_model.to(torch.device("cpu"))

                for i in range(len(gradients)):
                    gradients[i] = gradients[i] / float(len(task_batch))
                for i in range(len(q_gradients)):
                    q_gradients[i] = q_gradients[i] / float(len(task_batch))

                cos_mean = 0
                for l in range(len(gradients) - 2):
                    gradients_ = gradients[l].view(gradients[l].shape[0], -1)
                    q_gradients_ = q_gradients[l].view(q_gradients[l].shape[0], -1)
                    cos_mean += torch.mean(F.cosine_similarity(gradients_, q_gradients_, dim=-1)) / (len(gradients) - 2)
                if cos_mean >= 0:
                    for i, params in enumerate(model.cm.parameters()):
                        params.grad = gradients[i] + q_gradients[i]
                else:
                    for i, params in enumerate(model.cm.parameters()):
                        params.grad = gradients[i]

                out_opti.step()
                out_opti.zero_grad()

                scheduler.step()

                logger.append("{}, train_Step:{}, training Acc:{:.4f}, cos:{:.6f}".format(
                    datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), step, numpy.mean(task_acc), cos_mean))

                # valuation
                weight_after = OrderedDict((name, param) for (name, param) in model.state_dict().items())

                if args.do_valuation and step != 0 and (step+1) % 25 == 0:
                    val_model = MaskedLMTransformerClassification(args, device)
                    val = MetaTask(val_data,
                                   num_task=args.num_task_val,
                                   way=args.way,
                                   shot=args.shot,
                                   query=args.query,
                                   tokenizer=tokenizer,
                                   training=False,
                                   valuation=True,
                                   args=args)
                    batch_of_val_tasks = create_batch_of_tasks(val, batch_size=args.num_task_val, is_shuffle=False)

                    val_acc = []
                    for val_task_id, val_task in enumerate(batch_of_val_tasks):
                        val_model.load_state_dict(weight_after)
                        val_model.to(device)

                        val_support = val_task[0][0]
                        val_query = val_task[0][1]
                        val_support_dataloader = DataLoader(val_support, batch_size=len(val_support))

                        if args.optim == 'Adamw':
                            cm_optimizer = torch.optim.AdamW(val_model.parameters(),
                                                             lr=args.lr,
                                                             betas=(0.9, 0.99),
                                                             eps=1e-6,
                                                             weight_decay=0.1)
                        if args.optim == 'Adam':
                            cm_optimizer = torch.optim.Adam(val_model.parameters(), lr=args.lr)

                        val_model.train()
                        for i in range(args.num_update_step):
                            for val_sup_step, val_batch in enumerate(val_support_dataloader):
                                cm_optimizer.zero_grad()
                                val_batch = tuple(t.to(device) for t in val_batch)
                                input_ids, attention_mask, classification_label_id, lm_label_ids = val_batch
                                cm_loss, lm_loss, _ = val_model(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             classification_label_id=classification_label_id,
                                                             lm_label_ids=lm_label_ids,
                                                             pos1=None,
                                                             pos2=None)
                                train_loss = cm_loss * (1 - args.rho) + lm_loss * args.rho
                                train_loss.backward()

                                cm_optimizer.step()
                                logger.append("Epoch:{}, Number:{}, val_loss:{:.6f}".format(epoch, count, train_loss))

                        val_model.eval()
                        with torch.no_grad():
                            val_query_dataloader = DataLoader(val_query, sampler=None, batch_size=len(val_query))
                            val_query_batch = iter(val_query_dataloader).next()
                            val_query_batch = tuple(t.to(device) for t in val_query_batch)
                            input_ids, attention_mask, classification_label_id, lm_label_ids = val_query_batch
                            cm_loss, lm_loss, logits = val_model(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             classification_label_id=classification_label_id,
                                                             lm_label_ids=lm_label_ids,
                                                             pos1=None,
                                                             pos2=None)
                            cm_logits = F.softmax(logits, dim=1)
                            p_label_ids = torch.argmax(cm_logits, dim=1)
                            p_label_ids = p_label_ids.detach().cpu().numpy().tolist()
                            labels_ids = classification_label_id.detach().cpu().numpy().tolist()
                            # print(p_label_ids)

                            accuracy = accuracy_score(labels_ids, p_label_ids)
                            val_acc.append(accuracy)

                    logger.append("Epoch:{}, Val_task_id:{}, Task accuracy:{:.4f}".format(
                        epoch, val_task_id, numpy.mean(val_acc)))
                        # val_model.to(torch.device("cpu"))

                    cur_acc = numpy.mean(val_acc)

                    logger.append("{}, The total val_accuracy:{:.4f}, The best accuracy:{:.4f}".format(
                        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), cur_acc, best_acc))

                    if cur_acc > best_acc:
                        best_acc = cur_acc
                        if args.store_results:
                            best_model = model.state_dict()
                            torch.save(best_model, output_dir + save_path)
                            print("{} Save cur best model to {}".format(
                                 datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), args.output_dir))
                        cycle = 0
                    else:
                        cycle += 1
                    print("-" * 20, "Number:", count, "Valuation Ending", "-" * 20)
                    count += 1

            if cycle == 25:
                print("Valuation didn't improve, then we take early-stopping!!!")
                break

        print("{} End of training. Restore the best weights".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')))

    if args.do_test:
        print("-" * 20, "Testing Starting", "-" * 20)
        test_loss_list = []
        test_acc_list = []
        test = MetaTask(test_data,
                        num_task=args.num_task_test,
                        way=args.way,
                        shot=args.shot,
                        query=args.query,
                        tokenizer=tokenizer,
                        training=False,
                        valuation=False,
                        args=args)

        batch_of_test_tasks = create_batch_of_tasks(test, batch_size=1, is_shuffle=False)
        test_acc = []
        all_test_loss = []
        for test_id, test_task in enumerate(batch_of_test_tasks):
            if args.use_saving_model:
                model_dic = torch.load(output_dir + save_path, map_location=device)
            else:
                model_dic = OrderedDict((name, param) for (name, param) in model.state_dict().items())
            model.load_state_dict(model_dic)
            test_model = model

            test_support = test_task[0][0]
            test_query = test_task[0][1]
            test_support_dataloader = DataLoader(test_support, batch_size=len(test_support))

            if args.optim == 'Adamw':
                test_optimizer = torch.optim.AdamW(test_model.parameters(),
                                                   lr=args.lr,
                                                   betas=(0.9, 0.99),
                                                   eps=1e-6,
                                                   weight_decay=0.1)
            if args.optim == 'Adam':
                test_optimizer = torch.optim.Adam(test_model.parameters(), lr=args.lr)
            test_model.train()
            for i in range(args.num_update_step * 4):
                for test_sup_step, test_batch in enumerate(test_support_dataloader):
                    test_optimizer.zero_grad()
                    test_batch = tuple(t.to(device) for t in test_batch)
                    input_ids, attention_mask, classification_label_id, lm_label_ids = test_batch
                    cm_loss, lm_loss, _ = test_model(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     classification_label_id=classification_label_id,
                                                     lm_label_ids=lm_label_ids,
                                                     pos1=None, pos2=None)
                    if args.test_lm_loss:
                        test_loss = cm_loss * (1 - args.rho) + lm_loss * args.rho
                    else:
                        test_loss = cm_loss + lm_loss * 0
                    all_test_loss.append(test_loss.item())
                    test_loss.backward()
                    test_loss_list.append(numpy.mean(all_test_loss))
                    test_optimizer.step()

            test_model.eval()
            with torch.no_grad():
                test_query_dataloader = DataLoader(test_query, sampler=None, batch_size=len(test_query))
                test_query_batch = iter(test_query_dataloader).next()
                test_query_batch = tuple(t.to(device) for t in test_query_batch)
                input_ids, attention_mask, classification_label_id, lm_label_ids = test_query_batch
                cm_loss, lm_loss, logits = test_model(input_ids=input_ids,
                                                  attention_mask=attention_mask,
                                                  classification_label_id=classification_label_id,
                                                  lm_label_ids=lm_label_ids,
                                                  pos1=None, pos2=None)
                cm_logits = F.softmax(logits, dim=1)
                p_label_ids = torch.argmax(cm_logits, dim=1)
                p_label_ids = p_label_ids.detach().cpu().numpy().tolist()
                labels_ids = classification_label_id.detach().cpu().numpy().tolist()

                accuracy = accuracy_score(labels_ids, p_label_ids)
                test_acc.append(accuracy)
                test_acc_list.append(numpy.mean(test_acc))
            logger.append("{}, test_id:{}, test accuracy in test batch:{}".format(
                datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), test_id, numpy.mean(test_acc)))
        logger.append("all accuracy in test dataset:{:.4f}".format(numpy.mean(test_acc)))

