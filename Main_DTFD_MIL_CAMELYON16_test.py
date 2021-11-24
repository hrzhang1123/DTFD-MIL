import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
import random
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric

testMask_dir = '/home/sky/SSD_4TB/dataset/Camelyon16_Original/testing/mask'

parser = argparse.ArgumentParser(description='abc')

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)

parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log2', type=str)

parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)

parser.add_argument('--isCuda', default=True, type=bool)
parser.add_argument('--batch_size', default=1, type=int)   ##### 16
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)

parser.add_argument('--mDATA_dir_test0', default='/home/sky/SSD/Project/MIL_Camelyon_/train_features/dataset/ImageNet_Features/ResNet50D1024_L1Size256/mDATA_test.pkl', type=str)
parser.add_argument('--load_model_dir', default='/home/sky/myWork/myProject/MIL_General/save_model/model.pth', type=str)

parser.add_argument('--numGroup', default=5, type=int)
parser.add_argument('--total_instance', default=5, type=int)

parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)  ###
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--num_MeanInference', default=1, type=int)


def main():
    params = parser.parse_args()
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))
    in_chn = 1024

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        attCls = torch.nn.DataParallel(attCls)

    if params.load_model_dir != '':
        load_dict = torch.load(params.load_model_dir)
        classifier.load_state_dict(load_dict['classifier'])
        attention.load_state_dict(load_dict['attention'])
        dimReduction.load_state_dict(load_dict['dim_reduction'])
        attCls.load_state_dict(load_dict['att_classifier'])

    if params.isCuda:
        ce_cri = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    else:
        ce_cri = torch.nn.CrossEntropyLoss(reduction='none')

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log.txt')

    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))

    log_file = open(log_dir, 'a')

    with open(params.mDATA_dir_test0, 'rb') as f:
        mDATA_test = pickle.load(f)

    SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_test(mDATA_test)

    tauc = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri, epoch=-1,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, total_instance=params.total_instance)


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, criterion=None,  params=None, f_log=None, writer=None, numGroup=3, total_instance=3):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        SlideNames, FeatLists, Label = mDATA_list
        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ FeatLists[sst].to(params.device) for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                #tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
                        topk_idx_max = sort_idx[:instance_per_group].long()
                        topk_idx_min = sort_idx[-instance_per_group:].long()
                        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                        d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                        slide_d_feat.append(d_inst_feat)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

    print_log(f' Epoch {epoch}, First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
    print_log(f' Epoch {epoch}, Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)

    return auc_1

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def print_log(tstr, f):
    f.write('\n')
    f.write(tstr)
    print(tstr)

def reOrganize_mDATA_test(mDATA):

    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        print(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0

        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA(mDATA):

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('no such define slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label

if __name__ == "__main__":
    main()