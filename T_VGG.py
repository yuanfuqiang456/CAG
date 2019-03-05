#coding:utf8
from config import opt
import os
import torch as t
import models
from data.CAG import CAG
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.logger import Logger
import torch
import torchvision.models as Pre_models

logger = Logger(opt.log_path)

def test(**kwargs):
    opt.parse(kwargs)
    # configure model
    model = Pre_models.vgg19_bn(pretrained=True, num_classes=2)

    # data
    train_data = CAG(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()
        
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)

    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)
    
def train(**kwargs):
    opt.parse(kwargs)

    # step1: configure model
    #model = getattr(models, opt.model)()
    model = Pre_models.vgg19_bn(pretrained=True);
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(512 * 7 * 7, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 2),
        )
    #model = Pre_models.resnet152(pretrained=True);
    #model.classifier = torch.nn.Linear(2208, 2);
    #print(model)

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    #if opt.load_model_path:
    #    model.load(opt.load_model_path)
    #if opt.use_gpu: model.cuda()

    # step2: data
    train_data = CAG(opt.train_data_root,train=True)
    val_data = CAG(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)
    
    # step3: criterion and optimizer
    loss_func = t.nn.CrossEntropyLoss()

    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)

    previous_loss = 1.0

    # train
    for epoch in range(opt.max_epoch):
        print('epoch {}'.format(epoch + 1))
        train_num = 1
        train_acc = 0
        sum_loss = 0
        batch_num = 0
        for ii,(data,label) in enumerate(train_dataloader):
            # train model 
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            probability = t.nn.functional.softmax(score,dim=1)
            _, result = torch.max(probability, 1)
            train_correct = (result == target.squeeze(0)).sum()
            train_acc += train_correct.item()
            train_num += target.size(0)
            loss = loss_func(score,target)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()
            batch_num += 1


        print("当前loss:", sum_loss / batch_num)
        logger.scalar_summary('train_loss',sum_loss / batch_num, epoch)
        accuracy = train_acc / train_num
        logger.scalar_summary('train_accurancy', accuracy, epoch)

        if (epoch+1)%100==0:
        #if loss.item() < previous_loss:
            #model.save()
            lr = lr * opt.lr_decay
            print("当前学习率",lr)
            logger.scalar_summary('lr', lr, epoch)
            for param_group in optimizer.param_groups:#optimizer通过param_group来管理参数组. 通过更改param_group[‘lr’]的值来更改对应参数组的学习率。
                param_group['lr'] = lr
        #previous_loss = loss.item()

        # validate and visualize
        if (epoch+1)%5 == 0:
            val_accuracy,val_loss = val(model, val_dataloader)
            print("验证集上准确率为：", val_accuracy)
            logger.scalar_summary('val_accurancy', val_accuracy, epoch)
            print("val_loss:", val_loss)
            logger.scalar_summary('val_loss', val_loss, epoch)


def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    # 把模型设为验证模式
    model.eval()
    train_acc = 0
    val_num = 1
    loss_func = t.nn.CrossEntropyLoss()
    sum_loss = 0
    batch_num = 0
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        probability = t.nn.functional.softmax(score,dim=1)
        _, predicted = torch.max(probability, 1)
        loss = loss_func(probability, val_label)
        sum_loss += loss.item()
        train_correct = (predicted == val_label.squeeze(0)).sum()
        train_acc += train_correct.item()
        val_num += val_label.size(0)
        batch_num += 1

        # 把模型恢复为训练模式
    loss = sum_loss / batch_num
    model.train()
    accuracy = train_acc/val_num
    return  accuracy,loss

def help():
    '''
    打印帮助的信息： python file.py help
    '''
    
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    train();
#    import fire
    #train()
