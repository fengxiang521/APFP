import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate


class ProtoNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def feature_forward(self, x):
        out = self.avgpool(x).view(x.size(0),-1)
        return out

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        if self.n_support>1:
            scores1, new_scores1, new_scores2, new_scores3, new_scores4, PC_scores1, PC_scores2 = self.setO(z_support,z_query)
            return new_scores1, scores1, new_scores2
        else:
            scores = self.euclidean_dist(z_query, z_proto)
            return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores,scores2,scores3 = self.set_forward(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        topk_scores2, topk_labels2 = scores.data.topk(1, 1, True, True)
        topk_ind2 = topk_labels2.cpu().numpy()
        top1_correct2 = np.sum(topk_ind2[:, 0] == y_label)
        topk_scores3, topk_labels3 = scores.data.topk(1, 1, True, True)
        topk_ind3 = topk_labels3.cpu().numpy()
        top1_correct3 = np.sum(topk_ind3[:, 0] == y_label)
        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score

    def setO(self,supports,querys):
        T1 = supports.contiguous().unsqueeze(0).expand(self.n_way*self.n_query, -1, -1,-1)
        T2 = querys.contiguous().unsqueeze(1).expand(-1, self.n_support, -1)
        T2=T2.contiguous().unsqueeze(1).expand(-1, self.n_support,-1,-1)
        query=querys.contiguous().unsqueeze(1).expand(-1, 5, -1)
        z_proto = supports.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_proto=z_proto.contiguous().unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        Pscores=-(torch.pow(query-z_proto,2).sum(2))
        dist = torch.pow(T1 - T2, 2).sum(3)
        score = -dist
        softmax_output=F.softmax(score,dim=2)
        temperature = 0.007
        softmax_output1=F.softmax(score*temperature,dim=2)
        temperature = 0.008
        softmax_output2=F.softmax(score*temperature,dim=2)
        temperature = 0.01
        softmax_output3=F.softmax(score*temperature,dim=2)
        temperature = 0.005
        softmax_output4=F.softmax(score*temperature,dim=2)
        proto2=torch.sum(T1*softmax_output1.unsqueeze(3),dim=2)
        new_scores1=torch.pow(proto2-query,2).sum(2)

        new_scores2=torch.pow(torch.sum(T1*softmax_output2.unsqueeze(3),dim=2)-query,2).sum(2)
        new_scores3=torch.pow(torch.sum(T1*softmax_output3.unsqueeze(3),dim=2)-query,2).sum(2)

        new_scores4=torch.pow(torch.sum(T1*softmax_output4.unsqueeze(3),dim=2)-query,2).sum(2)

        PCproto1,PCproto2,weight2=self.setProto(supports)
        PCprotox1=PCproto1.unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        PCprotox2=PCproto2.unsqueeze(0).expand(self.n_way*self.n_query, -1, -1)
        PCscores1=-(torch.pow(PCprotox1-query,2).sum(2))
        PCscores2=-(torch.pow(PCprotox2-query,2).sum(2))
        sum_dim_3 = torch.abs(dist-torch.mean(dist, dim=2,keepdim=True))
        percentage_tensor = sum_dim_3 / torch.sum(sum_dim_3,dim=2,keepdim=True)
        temperature=0.0001
        weight = F.softmax(-percentage_tensor * temperature, dim=2)
        new_scores1 = torch.pow(torch.sum(T1 * ((softmax_output2+weight)/2).unsqueeze(3), dim=2) - query, 2).sum(2)

        return  Pscores,-new_scores1,-new_scores2,-new_scores3,-new_scores4,PCscores1,PCscores2

    def setProto(self,support):
        # 获取张量的维度
        #batch_size, num_samples, feature_dim = support.shape

        # 扩展张量，使其在第二个维度上与其他样本相减
        expanded_tensor1 = support.unsqueeze(1).expand(-1, 5, -1, -1)
        expanded_tensor2 = support.unsqueeze(2).expand(-1, -1, 5, -1)


        # 计算差值的绝对值
        abs_diff = torch.abs(expanded_tensor1 - expanded_tensor2).sum(3).sum(2)/4

        # 沿特征维度求和
        dist=torch.pow(expanded_tensor1-expanded_tensor2,2).sum(3).sum(2)/4
        temp = 0.004
        weight2=F.softmax(-dist*temp)
        proto1=torch.sum(support*weight2.unsqueeze(2),dim=1)
        temp=0.005
        weight2 = F.softmax(-dist * temp)
        proto2 = torch.sum(support * weight2.unsqueeze(2), dim=1)


        return proto1,proto2,weight2