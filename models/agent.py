import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv2DBatchNormRelu, deconv2DBatchNormRelu
from .backbone import resnet_encoder, simple_decoder, simple_classifier, n_segnet_decoder
from .sinkhorn_gpu import sinkhorn


################################################
#                  Modules                     #
################################################
class img_encoder(nn.Module):
    def __init__(self, n_classes=11, in_channels=3, feat_channel=512):
        super(img_encoder, self).__init__()

        self.feature_backbone = resnet_encoder(n_classes=n_classes, in_channels=in_channels)
        self.squeezer = conv2DBatchNormRelu(512, feat_channel, k_size=3, stride=1, padding=1)

    def forward(self, inputs):
        outputs = self.feature_backbone(inputs)
        outputs = self.squeezer(outputs)
        return outputs


class query_net(nn.Module):
    def __init__(self):
        super(query_net, self).__init__()

        # Encoder
        # down 1
        self.conv1 = conv2DBatchNormRelu(512, 256, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(256, 128, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(128, 32, k_size=3, stride=1, padding=1)

    def forward(self, features_map):
        outputs = self.conv1(features_map)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)  # (bsize, 32, 4, 4)
        return outputs


class key_net(nn.Module):
    def __init__(self):
        super(key_net, self).__init__()

        # Encoder
        # down 1
        self.conv1 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(512, 512, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(512, 1024, k_size=3, stride=1, padding=1)

    def forward(self, features_map):
        outputs = self.conv1(features_map)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)  # (bsize, 32, 4, 4)
        return outputs


#####################################################################
# our model
# called in train.py
#####################################################################
class MIMOcom(nn.Module):
    def __init__(self, n_classes=11, n_way=3, n_shot=1, n_query=9,
                 in_channels=3, feat_channel=512, feat_squeezer=-1,
                 agent_num=5, shuffle_flag=False, image_size=512, mode='meta',
                 solver='sinkhorn', key_size=128, query_size=128, is_seg=False, miter=10,
                 sfc_lr=0.1, sfc_wd=0, sfc_update_step=10, sfc_bs=4, n_head=1, max_weight=0):

        super(MIMOcom, self).__init__()

        self.agent_num = agent_num
        self.in_channels = in_channels
        self.shuffle_flag = shuffle_flag
        self.feature_map_channel = 512
        self.key_size = key_size
        self.query_size = query_size
        self.solver = solver
        self.n_way = n_way
        if n_shot > 1:
            print('Check SFC for using 1-shot learning')
        self.n_shot = n_shot
        self.n_proto = self.n_way  # self.n_way * self.n_shot
        self.n_query = n_query
        self.temperature = 12.5
        self.is_seg = is_seg
        self.miter = miter
        self.sfc_lr = sfc_lr
        self.sfc_wd = sfc_wd
        self.sfc_update_step = sfc_update_step
        self.sfc_bs = sfc_bs
        self.n_head = n_head
        self.max_weight = max_weight

        print('Construct MIMOcom =========')  # our model: detach the learning of values and keys

        self.encoder = img_encoder(n_classes=n_classes, in_channels=in_channels)

        # if mode=='meta':
        # generate matching key and query features
        self.key_net = key_net()  # 1024
        self.query_net = query_net()  # 32

        self.Wg = nn.Parameter(torch.FloatTensor(self.key_size, self.query_size), requires_grad=True)
        self.Wg.data.normal_(0.0, 0.01)

        # if n_head>1:
        #     self.Wg2 = nn.Parameter(torch.FloatTensor(n_head, self.key_size, self.query_size), requires_grad=True)
        #     self.Wg2.data.normal_(0.0, 0.01)

        # print('Msg size: ', query_size, '  Key size: ', key_size)
        self.softmax = nn.Softmax(dim=1)

        # Segmentation decoder
        # for pre-training, there is total of 11 classes
        if self.is_seg:
            # conv_trans upsamples input blob from (512,8,8) to (128,16,16)
            self.conv_trans = n_segnet_decoder(in_channels=self.feature_map_channel)
            # decoder head is not shared, it upsamples (128,16,16) to mask (n_way,512,512)
            if mode == 'meta':
                self.decoder = simple_decoder(n_classes=self.n_way + 1, in_channels=128)
            else:
                self.pre_decoder = simple_decoder(n_classes=n_classes, in_channels=128)
        else:
            self.pre_decoder = simple_classifier(n_classes=n_classes, in_channels=self.feature_map_channel)

    def compute_attention(self, qu, k):
        query = self.attention_weight(qu)  # (2,6,1024)
        prod = torch.bmm(k, query)  # (2,6,6)
        return prod

    # A is support, B is query
    # A should be 1024 x 4 x 4, B should be 32 x 4 x 4
    def get_source_dst(self, A, B, dir=True):

        # B size: (B,C,H,W)

        # [3, 1024, 4, 4], [27, 32, 4, 4]
        # print(A.size(), B.size(),'----1')  # [75, 640, 5, 5] [5, 640, 5, 5]
        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])  # (B,C,1,1)
        B = B.repeat(1, 1, A.shape[2], A.shape[3])  # (B,C,H,W)

        # transpose
        A = A.permute(0, 2, 3, 1).contiguous()
        B = B.permute(0, 2, 3, 1).contiguous()

        # print(A.size(), B.size())
        if dir:
            X = A
            G = torch.matmul(B, self.Wg)
            X = X.unsqueeze(1)
            G = G.unsqueeze(0)
            combination = (X * G).sum(-1)

        else:
            G = torch.matmul(A, self.Wg)
            X = B
            G = G.unsqueeze(1)
            X = X.unsqueeze(0)
            combination = (G * X).sum(-1)


        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    # use max_weight equation to compute node weights
    def get_source_dst_max(self, A, B, dir=True):

        # B size: (B,C,H,W)

        # [3, 1024, 4, 4], [27, 32, 4, 4]
        # print(A.size(), B.size(),'----1')  # [75, 640, 5, 5] [5, 640, 5, 5]
        M = A.shape[0]
        N = B.shape[0]

        combination_max = None

        # B = F.adaptive_avg_pool2d(B, [1, 1])  # (B,C,1,1)
        # B = B.repeat(1, 1, A.shape[2], A.shape[3])  # (B,C,H,W)

        H = A.size(-2)
        W = A.size(-1)

        # transpose
        A = A.permute(0, 2, 3, 1).contiguous()
        B = B.permute(0, 2, 3, 1).contiguous()

        for i in range(H):
            for j in range(W):

                B0 = B[:, i:i + 1, j:j + 1, :]
                # print(B0.size())
                # print(A.size(), B.size())
                if dir:
                    X = A
                    G = torch.matmul(B0, self.Wg)
                    X = X.unsqueeze(1)
                    G = G.unsqueeze(0)
                    # print(X.size(),G.size(),'---')
                    combination = (X * G).sum(-1)

                else:
                    G = torch.matmul(A, self.Wg)
                    X = B0
                    G = G.unsqueeze(1)
                    X = X.unsqueeze(0)
                    # print(X.size(), G.size(), '===')
                    combination = (G * X).sum(-1)

                combination = combination.view(M, N, -1)
                # print(X.size(), G.size(), combination.size(), '----2') # [27, 3, 8, 8, 32], [27, 3, 8, 8, 32]
                if combination_max is None:
                    combination_max = combination
                else:
                    combination_max = torch.max(combination_max, combination)

        combination_max = F.relu(combination_max) + 1e-3
        return combination_max

    def get_similiarity_map(self, proto, query):

        # [3, 1024, 4, 4], [27, 32, 4, 4]
        # print(proto.size(), query.size(),'get similarity map---')

        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)  # [3, 1024, 16]
        proto = proto.view(proto.shape[0], proto.shape[1], -1)  # [27, 32, 16]

        # print(proto.size(), query.size(), 'get similarity map---222')
        proto = proto.permute(0, 2, 1)  # [3,  16, 1024]
        query = query.permute(0, 2, 1)  # [27, 16, 32]
        # print(torch.isnan(proto).any(), torch.isnan(self.Wg).any(), '---proto 111')
        # print(proto.size(), query.size(), 'get similarity map---222')
        proto = torch.matmul(proto, self.Wg)  # ([3, 16, 32]
        # print(proto.size(), query.size(), 'get similarity map---222')

        # print(torch.isnan(proto).any(),
        #       torch.isnan(query).any(),'--similarity_map proto query')

        query = query.unsqueeze(1).repeat([1, way, 1, 1])  # [27, 3, 16, 32]
        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])  # [27, 3, 16, 32]
        # print(proto.size(), query.size(), 'get similarity map---222')

        query = query.unsqueeze(-2)  # [27, 3, 64, 1, 32]
        proto = proto.unsqueeze(-3)  # [27, 3, 1, 64, 32]
        # https://pytorch.org/docs/stable/nn.functional.html#cosine-similarity
        similarity_map = F.cosine_similarity(query, proto, dim=-1)
        # print(similarity_map.size()) # [27, 3, 16, 16]

        return similarity_map

    def generate_keys(self, sup):
        sup_keys = self.key_net(sup)
        # support [3, 1024, 4, 4],  qr_vals [27, 32, 4, 4]
        sup_keys = self.normalize_feature(sup_keys)
        return sup_keys

    def generate_qrvs(self, qr):
        qr_vals = self.query_net(qr)
        # support [3, 1024, 4, 4],  qr_vals [27, 32, 4, 4]
        qr_vals = self.normalize_feature(qr_vals)
        return qr_vals

    def emd_forward_1shot(self, proto, query):
        # weight is the importance of each grid of 4x4
        # weight_1 is for query image importance
        # weight_2 is for source (support) image importance
        if self.max_weight:
            # use Karpathy max weight
            weight_1 = self.get_source_dst_max(query, proto, True)
            weight_2 = self.get_source_dst_max(proto, query, False)
        else:
            weight_1 = self.get_source_dst(query, proto, True)  # src, Eq(9) in paper
            weight_2 = self.get_source_dst(proto, query, False)  # dst, Eq(9) in paper, s replaced with d
        # print(weight_1.size(), weight_2.size(),'----weight') # [27, 3, 64], [3, 27, 64]

        # print(weight_1, weight_1.sum(2))
        # this matrix is for computing cost matrix
        # similarity_map size [27, 3, 16, 16]
        similarity_map = self.get_similiarity_map(proto, query)  # Eq(2)  cost = 1-similarity
        # print(torch.isnan(weight_2).any(),
        #       torch.isnan(similarity_map).any(),'--similarity_map')
        # print(weight_1.sum(-1),'---w sum')

        if self.solver == 'sinkhorn':
            flow_map, weights, affinity_map = self.get_sinkhorn_distance(similarity_map, weight_1, weight_2, reg=0.2,
                                                                         maxIter=self.miter)
        else:
            raise NotImplemented('Solver not implemented!')

        # assert (not torch.isnan(flow_map).any()) and (not torch.isnan(weights).any())
        # print(logits.size()) # (27,3)
        return flow_map, weights, affinity_map

    def recon_feat(self, proto, query, weights, plan):

        weights_softmax = self.softmax(weights)  # (27, 3)
        proto_new = proto.view(proto.size(0), proto.size(1), -1)

        weights_softmax = weights_softmax.unsqueeze(-1).unsqueeze(-1)
        proto_exp = proto_new.unsqueeze(0).unsqueeze(-2)  # [1, 3, 512, 1, 64]
        plan_all = plan.unsqueeze(2)  # [27, 3, 1, 64, 64]

        plan_feat_prod = plan_all * proto_exp  # [27, 3, 512, 64, 64]
        plan_feat_prod_sum = plan_feat_prod.sum(-1)  # [27, 3, 512, 64]
        plan_feat_prod_sum_w = weights_softmax * plan_feat_prod_sum  # [27, 3, 512, 64]
        plan_feat = plan_feat_prod_sum_w.sum(1)  # [27, 512, 64]
        recon_feat = plan_feat.view(plan_feat.size(0), plan_feat.size(1), 8, 8)

        return recon_feat

    # SFC - structured FC layer
    # this module finetunes SFC
    def get_sfc(self, support):

        support = support.squeeze(0)

        # init the proto
        SFC = support.view(self.n_shot, -1, support.size(1), support.shape[-2], support.shape[-1]).mean(
            dim=0).clone().detach()

        # make SFC parameters, and finetune it to search for the optimal "averaged" dummy category sample
        # see paper DeepEMD section 3.5, section 4.1
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # create label for finetune
        label_shot = torch.arange(self.n_way).repeat(self.n_shot)
        label_shot = label_shot.long().to(support.device)

        with torch.no_grad():
            qr_qrvs = self.generate_qrvs(support)
            qr_qrvs = qr_qrvs.detach()

        with torch.enable_grad():
            for k in range(0, self.sfc_update_step):
                rand_id = torch.randperm(self.n_way * self.n_shot).to(support.device)
                for j in range(0, self.n_way * self.n_shot, self.sfc_bs):
                    if j + self.sfc_bs > self.n_way * self.n_shot:
                        continue
                    selected_id = rand_id[j: min(j + self.sfc_bs, self.n_way * self.n_shot)]
                    batch_shot = qr_qrvs[selected_id, :]
                    batch_label = label_shot[selected_id]

                    sup_keys = self.generate_keys(SFC)
                    sup_keys = sup_keys.detach()

                    optimizer.zero_grad()
                    _, logits, _ = self.emd_forward_1shot(sup_keys, batch_shot)
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def forward(self, inputs, loop, mode='meta', training=True, is_seg=False, sup_masks=None):

        if mode == 'meta':
            if is_seg:

                if isinstance(inputs, list):
                    sup_feat, qr_feat = inputs
                else:
                    sup_feat, qr_feat = inputs[:self.n_proto, ...], inputs[self.n_proto:, ...]

                qr_qrvs = self.generate_qrvs(qr_feat)
                sup_keys = self.generate_keys(sup_feat)

                _, weights, weight_maps = self.emd_forward_1shot(sup_keys, qr_qrvs)  # [27, 3, 64]
                weight_maps = weight_maps.view(weight_maps.size(0), weight_maps.size(1), 8, 8)
                pred = F.interpolate(weight_maps, size=[512, 512], mode='bilinear', align_corners=False)
                # print(pred.size()) # [27, 3, 512, 512]
                return pred
            else:

                if isinstance(inputs, list):
                    sup_feat, qr_feat = inputs
                else:
                    sup_feat, qr_feat = inputs[:self.n_proto, ...], inputs[self.n_proto:, ...]

                qr_qrvs = self.generate_qrvs(qr_feat)
                sup_keys = self.generate_keys(sup_feat)
                # directly return EMD distance
                _, weights, _ = self.emd_forward_1shot(sup_keys, qr_qrvs)
                return weights

        elif mode == 'pre_train':
            return self.pre_train_forward(inputs, is_seg)

        elif mode == 'encode':
            # [30, 3, 512, 512]
            return self.encoder(inputs)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input, is_seg=False):
        # print(input.size())  # [30, 3, 512, 512]
        feat_maps = self.encoder(input)  # [30, 512, 16, 16]
        # print(torch.isnan(feat_maps).any())  # --------- detect nan
        if is_seg:
            feat_maps = self.conv_trans(feat_maps)

        # for segmentation, the size is [50, 11, 512, 512]
        # for classification, the size is [50, 11]
        pred = self.pre_decoder(feat_maps)
        # print(pred.size())
        return pred

    def normalize_feature(self, x):
        x = x - x.mean(1).unsqueeze(1)
        return x

    def get_sinkhorn_distance(self, similarity_map, weight_1, weight_2, reg, maxIter):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        # num_node = weight_1.shape[-1]

        cost_matrix = (1 - similarity_map)  # .detach()


        similarity_map_new = torch.zeros_like(similarity_map).float()
        flow_map = torch.zeros_like(similarity_map).float()
        if torch.cuda.is_available():
            similarity_map_new = similarity_map_new.cuda()
            flow_map = flow_map.cuda()

        for i in range(num_query):
            for j in range(num_proto):
                # OT
                w1 = weight_1[i, j, :]
                w2 = weight_2[j, i, :]
                w1 = (w1 / w1.sum().item())  # .detach()
                w2 = (w2 / w2.sum().item())  # .detach()

                # print(weight1, weight1.sum(), weight2.sum())
                _, flow = sinkhorn(w1, w2, cost_matrix[i, j, :, :], reg, numItermax=maxIter,
                                   cuda=torch.cuda.is_available())
                # print(similarity_map_new[i, j, :, :])

                similarity_map_new[i, j, :, :] = similarity_map[i, j, :, :] * flow
                flow_map[i, j, :, :] = flow

        # print(temperature, num_node)
        weights = similarity_map_new.sum(-1).sum(-1) * self.temperature  # [75,5]
        affinity_map = similarity_map_new.sum(-1) * self.temperature

        return flow_map, weights, affinity_map
