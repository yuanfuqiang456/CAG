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
from PIL import  Image
from torchvision import  transforms as T
import cv2
import numpy as np

logger = Logger(opt.log_path)
#amp_handle = amp.init(enabled=True, verbose=False)
def test(**kwargs):
    opt.parse(kwargs)
    # configure model
    model = getattr(models, opt.model)().eval()
    AlexNet =  Pre_models.AlexNet(pretrained=True)
    # 读取参数
    pretrained_dict = AlexNet.state_dict()
    model_dict = model.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)

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
    model = getattr(models, opt.model)()
    #sync_bn_model = apex.parallel.convert_syncbn_model(model)
    #print(model)
    #AlexNet = Pre_models.alexnet(pretrained=True)
    # 读取参数
    #pretrained_dict = AlexNet.state_dict()
    #model_dict = model.state_dict()
    # 将pretrained_dict里不属于model_dict的键剔除掉
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 更新现有的model_dict
    #model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    #model.load_state_dict(model_dict)

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
            score,_ = model(input)
            probability = t.nn.functional.softmax(score,dim=1)
            _, result = torch.max(probability, 1)
            train_correct = (result == target.squeeze(0)).sum()
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
        if(accuracy>0.95):
            torch.save(model.state_dict(),  '/home/hdc/yfq/CAG/checkpoints/AlexNet1.pth')
        logger.scalar_summary('train_accurancy', accuracy, epoch)

        if (epoch+1)%10==0:
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
        score,_ = model(val_input)
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

#可视化代码
def visual(**kwargs):
    opt.parse(kwargs)
    model = models.AlexNet();
    checkpoint = torch.load('/home/hdc/yfq/CAG/checkpoints/AlexNet1.pth')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model.load_state_dict(state_dict, False)
    fc_weight = checkpoint['module.classifier.weight']
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms0 = T.Compose([
        T.RandomResizedCrop(512)
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
    cv2.imwrite('/home/hdc/yfq/CAG/data/visual1/3.CAM.bmp', result)
    return 1

def returnCAM(feature_conv, weight_softmax, class_idx = 2):
    # generate the class activation maps upsample to 256x256
    size_upsample = (512, 512)
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