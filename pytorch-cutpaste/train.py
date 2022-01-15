# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
from sklearn.metrics import roc_curve, auc
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import resnet18,wide_resnet50_2

from dataset import MVTecAT, Repeat
from cutpaste import CutPasteNormal,CutpasteDisturb, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn,CutPasteScar
from model import ProjectionNet
from eval import eval_model
import math
from util import showImage
import random
import numpy as np
import time

import sys
from UNetSR import UNetD4
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '8' 

f = open('result_bottle.out','w')
sys.stdout = f

def seed_torch(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
seed_torch()

def run_training(data_type="screw",
                 model_dir="models",
                 epochs=200,
                 pretrained=True,
                 test_epochs=1,
                 freeze_resnet=20,
                 learning_rate=1e-2,
                 optim_name="SGD",
                 batch_size=8,
                 head_layer=8,
                 cutpate_type=CutpasteDisturb,
                 device = "cuda",
                 workers=8,
                 size=300):

    after_cutpaste_transform=transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))    
                                                    
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    train_transform.transforms.append(transforms.Resize((size,size)))
    #
    train_transform.transforms.append(CutpasteDisturb(transform = after_cutpaste_transform))

    train_data = MVTecAT("Data", data_type, transform = train_transform, size=size)
    dataloader = DataLoader(train_data, batch_size=4, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            pin_memory=True)
                            
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size,size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]))
    test_data_eval = MVTecAT("Data", data_type, size, transform = test_transform, mode="test")
    dataloader_test = DataLoader(test_data_eval, batch_size=8,shuffle=False, num_workers=16)

    # create Model:
    blur = torch.nn.Conv2d(in_channels=3,out_channels=3,kernel_size=5,stride=1,padding=2).cuda()
    SRNet = UNetD4().cuda()
    bn = torch.nn.BatchNorm2d(3,affine=True,track_running_stats=True)
    [{'params':blur.parameters()}]
    mse = torch.nn.MSELoss()
    resnet = resnet18(pretrained = True).cuda()
    
    optimizer = torch.optim.Adam([
                	{'params': blur.parameters(), 'lr':1e-4}, 
                	{'params': SRNet.parameters(), 'lr':1e-3}, 
   	            ])
    print("blurlr:",1e-4,'srlr:',1e-3)
    print("train_bn:","true","test_bn:","false")
    print("sigmoid:","false","train_batch:","4","test_batch:","8")
    print("cut_type","disb")
    
    
    for epoch in range(epochs):
        
        running_loss=0.0
        blur.train()
        SRNet.train()
        for i, data in enumerate(dataloader):
             
            oimg = data[0]#原图 (16,3,300,300)
            timg = data[1]#缺陷图(16,3,300,300)
            
            oimg = bn(oimg)
            timg = bn(timg)
            
            oimg = oimg.cuda()
            timg = timg.cuda()
            
            blur_oimg = blur(oimg)
            blur_timg = blur(timg)
        
            sr_oimg = SRNet(blur_oimg)
            sr_timg = SRNet(blur_timg)
        
            #loss1:原图和复原图尽可能相似，优化超分SRNET
            loss1 = 1-torch.cosine_similarity(resnet(oimg), resnet(sr_oimg),dim=1).mean()
            #loss2:复原缺陷图和原图尽可能相似，优化超分网络，帮助超分后的图像掩盖缺陷
            loss2 = 1-torch.cosine_similarity(resnet(oimg), resnet(sr_timg),dim=1).mean()
            #loss3：模糊后的原图和模糊后的缺陷图之间尽可能相似，帮助模糊缺陷
            loss3 = 1-torch.cosine_similarity(resnet(blur_oimg), resnet(blur_timg),dim=1).mean()
            #loss4：缺陷图和缺陷复原图尽可能远离
            loss4 = 1-torch.cosine_similarity(resnet(timg),resnet(sr_timg),dim=1).mean()

            loss = loss1 + loss2 + loss3 - loss4

            # regulize weights:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if i % 10 == 9:
                print('[%d, %5d] loss:%.4f' % (epoch+1, i+1, running_loss / 10))
                running_loss = 0.0

        if test_epochs > 0 and epoch % test_epochs == 0:
            blur.eval()
            SRNet.eval()
            labels = []
            scores = []
            with torch.no_grad():
                for x, label in dataloader_test:
                    #print(x.shape) (64,3,256,256)
                    oimg = x
                    #oimg = bn(oimg)
                    oimg = oimg.cuda()
                    label = label.cuda()

                    bimg = blur(oimg)
                    srimg = SRNet(bimg)
                    
                    ofeature = resnet(oimg)
                    srfeature = resnet(srimg)
                    
                    ofeature = torch.nn.functional.normalize(ofeature, p=2, dim=1)
                    srfeature = torch.nn.functional.normalize(srfeature, p=2, dim=1)
    
                    if epoch == 5:
                        for parameters in blur.parameters():
                            print(parameters)
                        showImage(oimg[0].cpu(),"gridoimg")
                        showImage(bimg[0].cpu(),"gridbimg")
                        showImage(srimg[0].cpu(),"gridsrimg")
                        sys.exit()
                    '''
                    num = oimg.shape[0]          
                    for i in range(num):
                        score = -torch.log(mse(oimg[i],srimg[i]))
                        scores.append(score.cpu())
                    '''
                    score = torch.cosine_similarity(ofeature, srfeature)
                    scores.append(score.cpu())
                    labels.append(label.cpu())
                #scores = torch.sigmoid(torch.tensor(scores))
                #scores = torch.tensor(scores)
                #scores = scores.tolist()
                labels = torch.cat(labels)
                scores = torch.cat(scores)
                
                fpr, tpr, _ = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                print(epoch,"eval_auc:", round(roc_auc,2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="bottle",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--epochs', default=200, type=int,
                        help='number of epochs to train the model , (default: 300)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')
    
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')
    
    parser.add_argument('--test_epochs', default=1, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet (default: 20)')
    
    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    parser.add_argument('--head_layer', default=2, type=int,
                    help='number of layers in the projection head (default: 2)')
    
    parser.add_argument('--variant', default="union", choices=['normal', 'scar', '3way', 'union'], help='cutpaste variant to use (dafault: "3way")')
    
    parser.add_argument('--cuda', default=True,
                    help='use cuda for training (default: False)')
    
    parser.add_argument('--workers', default=16, type=int, help="number of workers to use for data loading (default:8)")


    args = parser.parse_args()
    print(args)
    all_types = ['bottle',
             'cable',
             'capsule',
             'carpet',
             'grid',
             'hazelnut',
             'leather',
             'metal_nut',
             'pill',
             'screw',
             'tile',
             'toothbrush',
             'transistor',
             'wood',
             'zipper']
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    variant_map = {'normal':CutPasteNormal, 'scar':CutPasteScar, '3way':CutPaste3Way, 'union':CutPasteUnion}
    variant = variant_map[args.variant]
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    
    # create modle dir
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))

    for data_type in types:
        print(f"training {data_type}")
        run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learning_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=CutpasteDisturb,
                     workers=args.workers)
