from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
from utils import utils as utils
from torch.utils.data import DataLoader
import time
import torch.nn.utils as torchutils
from torch.autograd import Variable
from utils.logger import Logger
import os
import numpy as np
from sklearn import metrics # added by us
import matplotlib.pyplot as plt # added by us

plt.switch_backend('agg')

prop_sum = np.empty(1)
y_true = np.empty(1)
pred_sum = np.empty(1)


if __name__ == '__main__':

    # Hyper Parameters
    parser = argparse.ArgumentParser() # Added by us
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')# previus 16
    parser.add_argument('--test_batch_size', type=int, default=8, help='test batch size')# previus 16
    parser.add_argument('--time_depth', type=int, default=15, help='number of time frames in each video\audio sample')
    parser.add_argument('--workers', type=int, default=0, help='num workers for data loading')
    parser.add_argument('--print_freq', type=int, default=50, help='freq of printing stats')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--use_mcb', action='store_true', help='wether to use MCB or concat')
    parser.add_argument('--mcb_output_size', type=int, default=1024, help='the size of the MCB outputl')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    parser.add_argument('--arch', type=str, default='AV', help='which modality to train - Video\Audio\AV')
    parser.add_argument('--pre_train', type=str, default='', help='path to the pre-trained network')
    parser.add_argument('--name', type=str, default='Default Model', help='Name of saved model plot and data')
    args = parser.parse_args()
    print(args, end='\n\n')

    # create test dataset
    dataset = utils.import_dataset(args)

    test_dataset = dataset(DataDir='/home/cv12f23/data/test/', timeDepth = args.time_depth, is_train=False)

    # create the data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False,
        drop_last=True)

    # create optimizer and loss
    criterion = nn.CrossEntropyLoss().cuda()

    # import network
    net = utils.import_network(args)

    # init from a saved checkpoint
    if args.pre_train is not '':
        model_name = os.path.join('pre_trained',args.arch,args.pre_train)

        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(args.pre_train))
            checkpoint = torch.load(args.pre_train)
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre_train, checkpoint['epoch']))
        else:
            print('Couldn\'t load model from {}'.format(model_name))
    else:
        print('Training the model from scratch.')


    # perform test
    test_acc = utils.AverageMeter()
    test_loss = utils.AverageMeter()

    net.eval()
    print('Test started.')

    all_pred = []
    all_gt = []

    for i, data in enumerate(test_loader):

        states_test = net.init_hidden(is_train=False)

        if args.arch == 'Video' or args.arch == 'Audio':  # single modality

            input, target = data  # input is of shape torch.Size([batch, channels, frames, width, height])
            input_var = Variable(input.unsqueeze(1)).cuda()
            target_var = Variable(target.squeeze()).cuda()

            output = net(input_var, states_test)

        else:  # multiple modalities

            audio, video, target = data
            audio_var = Variable(audio.unsqueeze(1)).cuda()
            video_var = Variable(video.unsqueeze(1)).cuda()
            target_var = Variable(target.squeeze()).cuda()

            output = net(audio_var, video_var, states_test)

        loss = criterion(output.squeeze(), target_var)

        # measure accuracy and record loss
        prop, predicted = torch.max(output.data, 1)
        #print(predicted)
        accuracy = (predicted == target.squeeze().cuda()).sum().type(torch.FloatTensor)
        #print(prop)
        #print(prop,target.squeeze().cuda())
        #fpr, tpr, thresholds = metrics.roc_curve(target.squeeze().cuda().cpu(),prop.cpu())
        #print(fpr,tpr,thresholds)
        prop_sum = np.concatenate((prop_sum,prop.cpu())) 
        y_true = np.concatenate((y_true,target.squeeze().cuda().cpu())) 
        pred_sum = np.concatenate((pred_sum, predicted.squeeze().cuda().cpu()))
        accuracy.mul_((100.0 / args.test_batch_size))
        test_loss.update(loss.item(), args.test_batch_size)
        test_acc.update(accuracy.item(), args.test_batch_size)
        #print(prop_sum)
        #print(y_true)
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]'.format(i, len(test_loader)))

    print('Test finished.')
    print('final loss on test set is {} and final accuracy is {}'.format(test_loss.avg,test_acc.avg))
    
    
    
    
    
    ### Code added by Oliver and Claes to compute metrics for different models ###
    
    
    print(y_true.astype(int))
    print(prop_sum)
    #metrics.RocCurveDisplay.from_predictions(y_true.astype(int),prop_sum)
    #plt.savefig("/home/cv12f23/ROC.svg")
    #plt.show()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #for i in range(1):
    fpr, tpr, _ = metrics.roc_curve(y_true.astype(int), prop_sum)
    roc_auc = metrics.auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    #fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_true.astype(int).ravel(), prop_sum.ravel())
    #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plot_path = "/home/cv12f23/" + args.name + ".svg"
    plt.savefig(plot_path)
    #plt.show()
    
    
    ##############################################################################
    # Calculate F1-score
    f1_score = metrics.f1_score(y_true.astype(int), pred_sum.astype(int))
    
    # Calculate Precision 
    precision = metrics.precision_score(y_true.astype(int), pred_sum.astype(int))
    
    # Calculate Recall
    recall = metrics.recall_score(y_true.astype(int), pred_sum.astype(int))
    
    # Write file
    import csv 
    
    stat_path = "/home/cv12f23/" + args.name + ".csv" 

    with open(stat_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Score"])
        writer.writerow(["Accuracy", test_acc.avg])
        writer.writerow(["F1-score", f1_score])
        writer.writerow(["Precision", precision])
        writer.writerow(["Recall", recall])
        writer.writerow(["AUC", roc_auc])
    
    
    
    
    
    
    
    