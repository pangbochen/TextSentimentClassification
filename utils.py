# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.vocab import GloVe, FastText
import numpy as np
from sklearn.metrics import accuracy_score


def load_data(opt):
    device = 0 if opt.use_cuda else -1

    # use torchtext as dataloader
    # from torchtext
    # dataset is imdb
    # data.Field
    # torchtext 的使用方法见 https://zhuanlan.zhihu.com/p/31139113?edition=yidianzixun&utm_source=yidianzixun&yidian_docid=0IIxNSe7&yidian_s=&yidian_appid=
    # https://github.com/pytorch/text#data
    # lower=True 将数据转换为小写
    # include_lengths=True
    # fix_length : 将每条数据的长度进行补全， 使用pad_token
    # pad_token : 用于补全的字符，默认为"<pad>"
    # sequential : 表示为序列
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=opt.max_seq_len)
    LABEL = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # get datasets

    # build vocab, use Glove embedding
    # can also use FastText or AllenNLP
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train)

    # generate train_iter and test_iter
    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=opt.batch_size, device=device, repeat=False, shuffle=True)

    # update dataset information
    opt.label_size = len(LABEL.vocab)
    opt.vocab_size = len(TEXT.vocab)
    opt.embedding_dim = TEXT.vocab.vectors.size()[1] # 300 as default
    opt.embeddings = TEXT.vocab.vectors

    return train_iter, test_iter


def clip_gradient(optimizer, grad_clip):
    # https://pytorch.org/docs/stable/torch.html#torch.clamp
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param.requires_grad:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def evaluate(model, test_iter, opt):
    model.eval()
    accuracy = []
    for index, batch in enumerate(test_iter):
        text = batch.text[0] # for torchtext
        pred = model(text)
        prob, idx = torch.max(pred, 1)
        percision = (idx==batch.label).float().mean()

        if opt.use_cuda:
            accuracy.append(percision.data.cpu())
        else:
            accuracy.append(percision.data)
    model.train()
    return np.mean(accuracy) # return the mean data

def evaluate_mixed(model, clf, test_iter, opt):
    '''
    :param model: nn model extract text feature
    :param clf: svm classifier
    :param test_iter: dataset iter
    :param opt: option
    :return: accuracy
    '''
    model.eval()
    label_fetures = torch.LongTensor()
    global cum_tensor
    cum_tensor = torch.Tensor()
    for index, batch in enumerate(test_iter):
        if index > 2:  # for test use
            break
        text = batch.text[0] # for torchtext
        model(text)
        if opt.mix_model:
            label_fetures = torch.cat((label_fetures, batch.label))
    print(cum_tensor.size())
    if opt.use_cuda:
        test_X = cum_tensor.data.cpu().numpy()
        test_y = label_fetures.data.cpu().numpy()

    else:
        test_X = cum_tensor.data.numpy()
        test_y = label_fetures.data.numpy()
    print('Start prediction')
    print(test_X)
    print(test_X.size())
    pred_y = clf.predict(test_X)
    accuracy = accuracy_score(pred_y, test_y)

    model.train()
    return accuracy # return the mean data

def evaluate_mixed_f1(model, clf, test_iter, opt):
    '''
    :param model: nn model extract text feature
    :param clf: svm classifier
    :param test_iter: dataset iter
    :param opt: option
    :return: accuracy
    '''
    model.eval()
    label_fetures = torch.LongTensor()
    cum_tensor = torch.Tensor()

    for index, batch in enumerate(test_iter):
        text = batch.text[0] # for torchtext
        pred, festure_tensor = model(text) # festure_tensor (batch_size, festure_dim)
        cum_tensor = torch.cat((cum_tensor, festure_tensor))


    if opt.use_cuda:
        test_X = cum_tensor.data.cpu().numpy()
        test_y = label_fetures.data.cpu().numpy()
    else:
        test_X = cum_tensor.data.numpy()
        test_y = label_fetures.data.numpy()
    pred_y = clf.predict(test_X)
    accuracy = accuracy_score((pred_y, test_y))

    model.train()
    return accuracy # return the mean data

def validate(model, val_iter, criterion, opt):
    model.eval()
    loss_list = []
    for index, batch in enumerate(val_iter):
        text = batch.text[0] # for torchtext
        pred = model(text)

        loss = criterion(pred, batch.label)
        if opt.use_cuda:
            loss_data = loss.cpu().data
        else:
            loss_data = loss.data
        loss_list.append(loss_data)
    model.train()
    return np.mean(loss_list)
