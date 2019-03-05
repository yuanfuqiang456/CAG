#coding:utf8
from __future__ import division
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
import numpy as np
from PIL import  Image
from torchvision import  transforms as T
import cv2


logger = Logger(opt.log_path)

def test(**kwargs):
    opt.parse(kwargs)
    # configure model
    model = Pre_models.densenet201(pretrained=True, num_classes=2)

    # data
    train_data = CAG(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        
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
    #model = Pre_models.densenet121(pretrained=True);#过拟合
    #model = Pre_models.densenet121()
    #model = models.densenet169(pretrained=True)
    model = models.densenet201(pretrained=True)
    #model = Pre_models.densenet161(pretrained=True)
    model.classifier = torch.nn.Linear(1920, 2);

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
    optimizer = t.optim.SGD(model.parameters(),lr = lr)

    previous_loss = 1.0

    # train
    for epoch in range(opt.max_epoch):
        print('epoch {}'.format(epoch + 1))
        train_num = 1
        train_acc = 0
        batch_num = 0
        loss_sum = 0
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
            train_correct = (result == target).sum()
            train_acc += train_correct.item()
            train_num += target.size(0)
            loss = loss_func(probability,target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_num += 1

        print("当前loss:", loss_sum/batch_num)
        logger.scalar_summary('train_loss', loss_sum/batch_num, epoch)
        accuracy = train_acc / train_num
        logger.scalar_summary('train_accurancy', accuracy, epoch)

        if (epoch+1)%100==0:
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
            #prefix = '/home/hdc/yfq/CAG/checkpoints/Densenet121'
            #name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            if val_accuracy >=0.96 :
                #torch.save(model.state_dict(),  name)
                opt.flag = True
            if val_accuracy <0.96:
                opt.flag = False
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
    TP = 0 #（真阳性）被正确诊断为患病的病人。
    TN = 0 # （真阴性）被正确诊断为健康的健康人。
    FP = 0 #(假阳性）被错误诊断为患病的健康人。
    FN = 0 #（假阴性）被错误诊断为健康的病人。

    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input)
        val_label = Variable(label.type(t.LongTensor))
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        probability = t.nn.functional.softmax(score,dim=1)
        loss = loss_func(probability, val_label)
        sum_loss += loss.item()
        _, predicted = torch.max(probability, 1)#返回输入Tensor中每行的最大值，并转换成指定的dim（维度）
        TP += (val_label[predicted==1]==1).type(torch.cuda.FloatTensor).sum()
        TN += (val_label[predicted==0]==0).type(torch.cuda.FloatTensor).sum()
        FP += (val_label[predicted == 1] == 0).type(torch.cuda.FloatTensor).sum()
        FN += (val_label[predicted == 0] == 1).type(torch.cuda.FloatTensor).sum()
        train_correct = (predicted == val_label).sum()
        train_acc += train_correct.item()
        val_num += val_label.size(0)
        batch_num += 1

        # 把模型恢复为训练模式
    if opt.flag:
        print("敏感度为：")
        print(TP/(TP + FN))
        print("特异度为：")
        print(TN/(TN + FP))
        print("查准率为：")
        print( TP/(TP + FP))
        print("查全率为：")
        print(TP/(TP + FN))
    loss = sum_loss/batch_num
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



#可视化代码
def visual(**kwargs):
    opt.parse(kwargs)
    model = models.densenet121()
    model.classifier = torch.nn.Linear(1024, 2);
    checkpoint = torch.load('/home/hdc/yfq/CAG/checkpoints/Densenet1210219_19:09:46.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict and "classifier" not in k}
    model.load_state_dict(state_dict, False)

    fc_weight = checkpoint['module.classifier.weight']
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms0 = T.Compose([
        T.RandomResizedCrop(224)
    ])
    transforms1 = T.Compose([
        T.ToTensor(),
        normalize
    ])
    # data
    img_path = '/home/hdc/yfq/CAG/data/visual1/3.jpg'
    data = Image.open(img_path)
    data0 = transforms0(data)
    data1 = transforms1(data0)
    data1 = data1.unsqueeze(0)
    model.eval()
    score,feature = model(data1)
    CAMs = returnCAM(feature,fc_weight)
    _,_,height, width = data1.size()
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[1], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + np.array(data0) * 0.5
    cv2.imwrite('/home/hdc/yfq/CAG/data/visual1/3.CAM0.bmp', result)
    return 1

def returnCAM(feature_conv, weight_softmax, class_idx = 2):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    weight_softmax = weight_softmax.unsqueeze(1)
    for idx in range(class_idx):
        cam = weight_softmax[idx].cpu().mm(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w).detach().numpy()
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__=='__main__':
    #train();
    visual();