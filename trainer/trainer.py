import os
import time
import torch
import numpy as np
from tqdm import tqdm
from tools.metrics import runningScore, averageMeter
from tools.utils import count_acc, count_acc_mask, Averager
from tools.plot_confusion_matrix import get_precision_recall
import matplotlib.pyplot as plt
from loader.init_val import name2id
from itertools import cycle


class Trainer_MIMOcom(object):
    def __init__(self, cfg, args, logdir, logger, model, loss_fn, trainloader, valloader, optimizer, scheduler, device):

        self.cfg = cfg
        self.args = args
        self.logdir = logdir
        self.logger = logger
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_classes = 11
        self.loss_fn = loss_fn

        if args.is_seg:
            self.running_metrics_val = runningScore(self.n_classes)

        self.device = device
        self.save_interval = 2000

    def train(self):
        # load model
        print('LearnMIMOCom_Trainer in trainer.py -------------')
        start_iter = 0

        val_loss_meter = averageMeter()
        time_meter = averageMeter()

        best_iou = -100.0
        best_epoch = -1
        i = start_iter

        flag = True

        loop = self.args.loop

        ################################################
        # label for query set,
        # always in the same pattern, 012340123401234...
        ################################################
        # label = torch.arange(self.args.way, dtype=torch.int8).repeat(self.args.shot + self.args.query)
        if self.args.is_seg:
            pass
        else:
            # starting from 0, 1, 2
            label_fsl = torch.arange(self.args.way, dtype=torch.int8).repeat(self.args.query)
            label_fsl = label_fsl.to(self.device)  # [30]: 3*(1+9) = 30
            label_fsl = label_fsl.long()

        # training
        total_iter = self.cfg["training"]["train_iters"]
        ns = self.args.shot * self.args.way

        while i <= total_iter and flag:
            self.model.train()

            tl = Averager()
            ta = Averager()

            print('Iter %d/%d' % (i, total_iter))

            for data in self.trainloader:
                start_ts = time.time()

                if self.args.ph == 0:

                    if self.args.is_seg == 1:
                        ########################
                        # pre-train seg
                        # in normal way
                        ########################

                        images, labels, _, _, _ = data

                        images = images[:32, ...]
                        labels = labels[:32, ...]
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        # print(images.size(), labels.size(), torch.unique(labels))

                        # print(torch.unique(labels))

                        outputs = self.model(images, loop, mode='pre_train', is_seg=True)

                        self.optimizer.zero_grad()
                        loss = self.loss_fn(input=outputs, target=labels)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()





                    else:
                        # for pre-training classification
                        # we use masked images with one class left
                        # also use that class true label
                        _, _, image_rois, _, cls_labels = data
                        images = image_rois.to(self.device)
                        labels = cls_labels.to(self.device)
                        # print(cls_labels)

                        outputs = self.model(images, loop, mode='pre_train', is_seg=False)

                        # print(outputs.size(),'----')
                        acc = count_acc(outputs, labels)

                        # print(outputs.size(), action_argmax.size()) # (12,11,512,512), (2,6)
                        # print(action_argmax)
                        # image loss
                        self.optimizer.zero_grad()
                        loss = self.loss_fn(input=outputs, target=labels)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()
                        tl.add(loss.item())
                        ta.add(acc)


                else:

                    if self.args.is_seg == 1:
                        ######################
                        # segmentation
                        ######################
                        images, labels, _, _, cls = data

                        images_new = torch.zeros_like(images).float()
                        labels_new = torch.ones_like(labels).long() * 255

                        # map pixels to few-shot labels
                        fs_cls_to_real_cls = cls[:self.args.way].numpy().tolist()
                        # assert len(fs_cls_to_real_cls) == self.args.way

                        for j in range(ns):
                            r_cls = fs_cls_to_real_cls[j % self.args.way]
                            idx = labels[j, ...] == r_cls
                            labels_new[j, idx] = j % self.args.way
                            images_new[j, :, idx] = images[j, :, idx]

                        for j in range(ns, images.size(0)):
                            for ix, r_cls in enumerate(fs_cls_to_real_cls):
                                idx = labels[j, ...] == r_cls
                                labels_new[j, idx] = ix
                                images_new[j, :, idx] = images[j, :, idx]

                        images = images_new.to(self.device)
                        # labels_new = labels_new.to(self.device)
                        # sup_masks, labels = labels_new[:ns, ...], labels_new[ns:, ...]
                        labels = labels_new[ns:, ...].to(self.device)
                        feat_maps = self.model(images, loop, mode='encode', is_seg=True)

                        if self.args.shot > 1:
                            data_shot, data_query = feat_maps[:ns], feat_maps[ns:]
                            sfc_maps = self.model.get_sfc(data_shot)
                            outputs = self.model([sfc_maps, data_query], loop, mode='meta', is_seg=True)
                        else:
                            outputs = self.model(feat_maps, loop, mode='meta', is_seg=True)

                        #outputs = self.model(feat_maps, loop, mode='meta', is_seg=True)

                        # segmentation loss
                        self.optimizer.zero_grad()
                        loss = self.loss_fn(input=outputs, target=labels)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                        acc = count_acc_mask(outputs, labels, 3)
                        tl.add(loss.item())
                        ta.add(acc)

                    else:
                        ######################
                        # classification
                        ######################
                        _, _, image_rois, label_rois, _ = data
                        image_rois = image_rois.to(self.device)

                        feat_maps = self.model(image_rois, loop, mode='encode', training=True, is_seg=False)

                        if self.args.shot > 1:
                            data_shot, data_query = feat_maps[:ns], feat_maps[ns:]
                            sfc_maps = self.model.get_sfc(data_shot)
                            # [3, 512, 8, 8], [27, 512, 8, 8]
                            outputs = self.model([sfc_maps, data_query], loop, mode='meta', training=True, is_seg=False)
                        else:
                            outputs = self.model(feat_maps, loop, mode='meta', training=True, is_seg=False)
                        #outputs = self.model(feat_maps, loop, mode='meta', training=True, is_seg=False)

                        labels = label_fsl
                        acc = count_acc(outputs, labels)

                        # image loss
                        self.optimizer.zero_grad()
                        loss = self.loss_fn(input=outputs, target=labels)
                        loss.backward()
                        self.optimizer.step()
                        self.scheduler.step()

                        tl.add(loss.item())
                        ta.add(acc)

                time_meter.update(time.time() - start_ts)
                # Process display on screen
                if i % self.cfg["training"]["print_interval"] == 0:  # 50
                    # print(self.scheduler.get_last_lr())
                    fmt_str = "Iter [{:d}/{:d}] Lr: {:.6f} Loss: {:.4f} AvgLoss: {:.4f}, " \
                              "AvgACC: {:.4f}  Time/Image: {:.4f}"
                    print_str = fmt_str.format(
                        i,
                        self.cfg["training"]["train_iters"],  # 200000
                        self.scheduler.get_last_lr()[0],
                        loss.item(),
                        tl.item(), ta.item(),
                        time_meter.avg / self.cfg["training"]["batch_size"],  # 2
                    )

                    print(print_str)
                    self.logger.info(print_str)
                    time_meter.reset()

                ###  Validation
                if (i + 1) % self.cfg["training"]["val_interval"] == 0 or (i + 1) == self.cfg["training"][
                    "train_iters"]:  # 1000
                    self.model.eval()
                    vl = Averager()
                    va = Averager()

                    with torch.no_grad():
                        # for i_val, data_val_list in tqdm(enumerate(self.valloader)):
                        #     break

                        for i_val, data_val_list in tqdm(enumerate(self.valloader)):

                            # if i_val % 20 == 0:
                            #     print('[%d/%d] Loss %.4f, Acc %.4f' % (i, len(self.valloader), vl.item(), va.item()))

                            if self.args.ph == 0:
                                # pre-train
                                if self.args.is_seg == 1:

                                    images_val, labels_val, _, _, _ = data_val_list

                                    images_val = images_val[:32, ...]
                                    labels_val = labels_val[:32, ...]

                                    images_val = images_val.to(self.device)
                                    labels_val = labels_val.to(self.device)
                                    # perform task
                                    # print('pre-training')
                                    # print(np.unique(labels))

                                    outputs = self.model(images_val, loop=None, mode='pre_train', is_seg=True)
                                    gt = labels_val.data.cpu().numpy()

                                    val_loss = self.loss_fn(input=outputs, target=labels_val)
                                    pred = outputs.data.max(1)[1].cpu().numpy()
                                    # print(outputs.size(), gt.shape, pred.shape)

                                    self.running_metrics_val.update(gt, pred)
                                    val_loss_meter.update(val_loss.item())
                                else:
                                    ######################
                                    # classification
                                    ######################
                                    _, _, image_rois_val, _, label_true = data_val_list

                                    image_rois = image_rois_val.to(self.device)
                                    labels = label_true.to(self.device)

                                    outputs = self.model(image_rois, loop=None, mode='pre_train', is_seg=False)

                                    # print(labels, outputs)
                                    acc = count_acc(outputs, labels)
                                    loss = self.loss_fn(input=outputs, target=labels)

                                    vl.add(loss.item())
                                    va.add(acc)



                            else:

                                if self.args.is_seg == 1:

                                    images, labels, _, _, cls = data_val_list

                                    images_new = torch.zeros_like(images).float()
                                    labels_new = torch.ones_like(labels).long() * 255

                                    # map pixels to few-shot labels
                                    fs_cls_to_real_cls = cls[:self.args.way].numpy().tolist()
                                    assert len(fs_cls_to_real_cls) == self.args.way
                                    ns = self.args.shot * self.args.way

                                    for j in range(ns):
                                        r_cls = fs_cls_to_real_cls[j % self.args.way]
                                        idx = labels[j, ...] == r_cls
                                        labels_new[j, idx] = j % self.args.way
                                        images_new[j, :, idx] = images[j, :, idx]

                                    for j in range(ns, images.size(0)):
                                        for ix, r_cls in enumerate(fs_cls_to_real_cls):
                                            idx = labels[j, ...] == r_cls
                                            labels_new[j, idx] = ix
                                            # print(ix, r_cls)
                                            images_new[j, :, idx] = images[j, :, idx]

                                    images = images_new.to(self.device)
                                    labels_new = labels_new.to(self.device)
                                    # sup_masks, labels = labels_new[:ns, ...], labels_new[ns:, ...]
                                    labels = labels_new[ns:, ...]

                                    feat_maps = self.model(images, loop, mode='encode', is_seg=True)

                                    if self.args.shot > 1:
                                        data_shot, data_query = feat_maps[:ns], feat_maps[ns:]
                                        sfc_maps = self.model.get_sfc(data_shot)
                                        outputs = self.model([sfc_maps, data_query], loop, mode='meta', is_seg=True)
                                    else:
                                        outputs = self.model(feat_maps, loop, mode='meta', is_seg=True)

                                    # outputs = self.model(feat_maps, loop, mode='meta', is_seg=True)
                                    val_loss = self.loss_fn(input=outputs, target=labels)

                                    acc = count_acc_mask(outputs, labels, 3)
                                    vl.add(val_loss.item())
                                    va.add(acc)

                                    ################ Put this in test !!!!!!!!!!!!!!!!!!!!!!######################
                                    # gt = labels.data.cpu().numpy()
                                    # pred = outputs.data.max(1)[1]
                                    # pred[labels == 255] = 255
                                    # pred = pred.cpu().numpy()  # (27,512,512)
                                    #
                                    # # now we map fsl labels back to original labels
                                    # pred_new = np.ones_like(pred) * 255
                                    # gt_new = np.ones_like(gt) * 255
                                    # for j in range(pred.shape[0]):
                                    #     for ix, r_cls in enumerate(fs_cls_to_real_cls):
                                    #         idx = gt[j, ...] == ix
                                    #         gt_new[j, idx] = r_cls
                                    #         idx2 = pred[j, ...] == ix
                                    #         pred_new[j, idx2] = r_cls
                                    #
                                    # self.running_metrics_val.update(gt_new, pred_new)



                                else:
                                    ######################
                                    # classification
                                    ######################
                                    _, _, image_rois_val, label_rois_val, _ = data_val_list

                                    image_rois = image_rois_val.to(self.device)

                                    feat_maps = self.model(image_rois, loop=loop, mode='encode', is_seg=False)
                                    if self.args.shot > 1:
                                        data_shot, data_query = feat_maps[:ns], feat_maps[ns:]
                                        sfc_maps = self.model.get_sfc(data_shot)
                                        outputs = self.model([sfc_maps, data_query], loop=loop, mode='meta',
                                                             is_seg=False)
                                    else:
                                        outputs = self.model(feat_maps, loop=loop, mode='meta', is_seg=False)

                                    #outputs = self.model(feat_maps, loop=loop, mode='meta', is_seg=False)
                                    labels = label_fsl

                                    acc = count_acc(outputs, labels)
                                    loss = self.loss_fn(input=outputs, target=labels)

                                    vl.add(loss.item())
                                    va.add(acc)

                    if self.args.is_seg == 1:
                        print("Overall")
                        # score, class_iou = self.running_metrics_val.get_scores()
                        # self.running_metrics_val.print_score(score, class_iou)
                        # self.running_metrics_val.reset()

                        print('Iteration %d' % (i,))
                        print('Validation loss %.4f' % (vl.item()))
                        print('Validation acc %.4f' % (va.item()))

                        # store the best model
                        if va.item() >= best_iou:
                            best_epoch = i
                            best_iou = va.item()
                            state = {
                                "epoch": i,
                                "model_state": self.model.state_dict(),
                                "best_iou": best_iou,
                            }
                            save_path = os.path.join(self.logdir, "best_model.pth")
                            print('Found best in epoch %d' % (i,))
                            torch.save(state, save_path)
                            print('Save to %s' % (save_path,))
                        else:
                            print('Best was in epoch %d' % (best_epoch,))


                    else:
                        print('Iteration %d' % (i,))
                        print('Validation loss %.4f' % (vl.item()))
                        print('Validation acc %.4f' % (va.item()))

                        # store the best model
                        if va.item() >= best_iou:
                            best_epoch = i
                            best_iou = va.item()
                            state = {
                                "epoch": i,
                                "model_state": self.model.state_dict(),
                                "best_iou": best_iou,
                            }
                            save_path = os.path.join(self.logdir, "best_model.pth")
                            print('Found best in epoch %d' % (i,))
                            torch.save(state, save_path)
                            print('Save to %s' % (save_path,))

                        else:
                            print('Best was in epoch %d' % (best_epoch,))

                    self.model.train()

                # save models by iteration
                if i > 0 and i % self.save_interval == 0:
                    state = {
                        "epoch": i,
                        "model_state": self.model.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(self.logdir, "model_%d.pth" % (i // self.save_interval,))
                    print('Save to %s' % (save_path,))
                    torch.save(state, save_path)

                if i == self.cfg["training"]["train_iters"]:
                    flag = False
                    break

                i += 1

        return save_path

    def evaluate(self, testloader):

        assert self.args.ph == 1
        loop = self.args.loop

        if self.args.is_seg:
            pass
        else:
            # starting from 0, 1, 2
            label_fsl = torch.arange(self.args.way, dtype=torch.int8).repeat(self.args.query)
            label_fsl = label_fsl.to(self.device)  # [30]: 3*(1+9) = 30
            label_fsl = label_fsl.long()

        running_metrics = runningScore(self.n_classes)
        time_meter = averageMeter()

        vl = Averager()
        va = Averager()

        ap_scores = []
        ap_gts = []

        # Setup Model
        self.model.eval()
        self.model.to(self.device)

        ns = self.args.shot * self.args.way

        print('Support data number', ns)
        print('Query data number', self.args.query * self.args.way)

        id2name = {}
        for k, v in name2id.items():
            id2name[v] = k
        cat_names = [id2name[a] for a in self.args.split_label if a != 0]

        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        with torch.no_grad():
            for i, data_list in enumerate(testloader):

                if i>0 and i % 20 == 0:
                    print('[%d/%d] Loss %.4f, Acc %.4f, FPS %.3f' % (i, len(testloader), vl.item(), va.item(),
                                                                (self.args.query * self.args.way) / time_meter.avg))

                images_val, labels_val, image_rois_val, label_rois_val, cls = data_list

                if self.args.is_seg == 1:
                    ######################
                    # segmentation
                    ######################
                    images_new = torch.zeros_like(images_val).float()
                    labels_new = torch.ones_like(labels_val).long() * 255

                    # map pixels to few-shot labels
                    fs_cls_to_real_cls = cls[:self.args.way].numpy().tolist()
                    #assert len(fs_cls_to_real_cls) == self.args.way
                    ns = self.args.shot * self.args.way

                    for j in range(ns):
                        r_cls = fs_cls_to_real_cls[j % self.args.way]
                        idx = labels_val[j, ...] == r_cls
                        labels_new[j, idx] = j % self.args.way
                        images_new[j, :, idx] = images_val[j, :, idx]

                    for j in range(ns, images_val.size(0)):
                        for ix, r_cls in enumerate(fs_cls_to_real_cls):
                            idx = labels_val[j, ...] == r_cls
                            labels_new[j, idx] = ix
                            # print(ix, r_cls)
                            images_new[j, :, idx] = images_val[j, :, idx]

                    images = images_new.to(self.device)
                    labels_new = labels_new.to(self.device)
                    # sup_masks, labels = labels_new[:ns, ...], labels_new[ns:, ...]
                    labels = labels_new[ns:, ...]

                    start_ts = time.time()
                    feat_maps = self.model(images, loop, mode='encode', is_seg=True)

                    if self.args.shot > 1:
                        data_shot, data_query = feat_maps[:ns], feat_maps[ns:]
                        sfc_maps = self.model.get_sfc(data_shot)
                        outputs = self.model([sfc_maps, data_query], loop, mode='meta', is_seg=True)
                    else:
                        outputs = self.model(feat_maps, loop, mode='meta', is_seg=True)

                    #outputs = self.model(feat_maps, loop, mode='meta', is_seg=True)

                    val_loss = self.loss_fn(input=outputs, target=labels)

                    acc = count_acc_mask(outputs, labels, 3)
                    time_meter.update(time.time() - start_ts)

                    vl.add(val_loss.item())
                    va.add(acc)

                    gt = labels.data.cpu().numpy()
                    pred = outputs.data.max(1)[1]
                    pred[labels == 255] = 255
                    pred = pred.cpu().numpy()  # (27,512,512)

                    output_soft = torch.softmax(outputs, dim=1)
                    output_soft = output_soft.cpu().numpy()

                    # now we map fsl labels back to original labels
                    pred_new = np.ones_like(pred) * 255
                    gt_new = np.ones_like(gt) * 255
                    for j in range(pred.shape[0]):
                        for ix, r_cls in enumerate(fs_cls_to_real_cls):
                            idx = gt[j, ...] == ix
                            gt_new[j, idx] = r_cls
                            idx2 = pred[j, ...] == ix
                            pred_new[j, idx2] = r_cls

                            tmp_score0 = output_soft[j, :, idx]
                            tmp_lb = np.zeros((tmp_score0.shape[0], self.n_classes))
                            tmp_score = np.zeros((tmp_score0.shape[0], self.n_classes))
                            tmp_lb[:, r_cls] = 1
                            tmp_score[:, fs_cls_to_real_cls] = tmp_score0

                            if tmp_lb.shape[0] > 100:
                                perm = np.random.permutation(tmp_lb.shape[0])
                                perm = perm[:100]
                                tmp_lb = tmp_lb[perm, :]
                                tmp_score = tmp_score[perm, :]

                            # print(tmp_score.shape, tmp_lb.shape)
                            ap_gts.append(tmp_lb)
                            ap_scores.append(tmp_score)

                    running_metrics.update(gt_new, pred_new)

                else:
                    ######################
                    # classification
                    ######################
                    image_rois = image_rois_val.to(self.device)

                    start_ts = time.time()
                    feat_maps = self.model(image_rois, loop, mode='encode', is_seg=False)
                    if self.args.shot > 1:
                        data_shot, data_query = feat_maps[:ns], feat_maps[ns:]
                        #print(feat_maps.size(), data_query.size())
                        sfc_maps = self.model.get_sfc(data_shot)
                        outputs = self.model([sfc_maps, data_query], loop, mode='meta', is_seg=False)
                    else:
                        outputs = self.model(feat_maps, loop, mode='meta', is_seg=False)

                    #outputs = self.model(feat_maps, loop, mode='meta', training=False, is_seg=False)
                    labels = label_fsl

                    loss = self.loss_fn(input=outputs, target=labels)

                    acc = count_acc(outputs, labels)
                    time_meter.update(time.time() - start_ts)

                    vl.add(loss.item())
                    va.add(acc)

                    fs_cls_to_real_cls = cls[:self.args.way].numpy().tolist()
                    #assert len(fs_cls_to_real_cls) == self.args.way
                    gt = labels.data.cpu().numpy()
                    pred = torch.argmax(outputs, dim=1)
                    pred = pred.cpu().numpy()

                    output_soft = torch.softmax(outputs, dim=1)
                    output_soft = output_soft.cpu().numpy()

                    # now we map fsl labels back to original labels
                    pred_new = np.zeros_like(pred)
                    gt_new = np.zeros_like(gt)
                    for j in range(pred.shape[0]):
                        # assert gt[j] == j%3
                        idx = j % 3
                        gt_new[j] = fs_cls_to_real_cls[idx]
                        pred_new[j] = fs_cls_to_real_cls[pred[j]]
                        tmp_lb = np.zeros((1, self.n_classes))
                        tmp_score = np.zeros((1, self.n_classes))
                        tmp_lb[0, fs_cls_to_real_cls[idx]] = 1
                        for ix, rls in enumerate(fs_cls_to_real_cls):
                            tmp_score[0, rls] = output_soft[j, ix]
                        ap_gts.append(tmp_lb)
                        ap_scores.append(tmp_score)

                    running_metrics.update(gt_new, pred_new)

                # if i>2:
                #     break

            if self.args.is_seg == 1:
                print("Segmentation")
            else:
                print("Classification")

            score, class_iou = running_metrics.get_scores()
            running_metrics.print_score(score, class_iou, self.args.split_label)

            print('Test loss %.4f' % (vl.item()))
            print('Test acc %.4f' % (va.item()))
            print('Per-class acc %.4f' % (score["Mean Acc : \t"]))
            print('FPS', (self.args.query * self.args.way) / time_meter.avg)

            ap_gts = np.concatenate(ap_gts, 0)
            ap_scores = np.concatenate(ap_scores, 0)

            # trim ap_scores
            print(self.args.split_label)
            split_label = self.args.split_label[1:]  # trim 0
            ap_gts = ap_gts[:, split_label]
            ap_scores = ap_scores[:, split_label]
            precision, recall, average_precision = get_precision_recall(ap_gts, ap_scores)

            n_classes = ap_scores.shape[1]

            plt.figure(figsize=(8, 8))
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            legends = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

            lines.append(l)
            legends.append('iso-f1 curves')

            for i, color in zip(range(n_classes), colors):
                l, = plt.plot(recall[i], precision[i], color=color, lw=2)
                lines.append(l)
                legends.append('PR for %s (area = %0.2f)'
                               % (cat_names[i], average_precision[i]))

            aps = [average_precision[i] for i in range(n_classes)]

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=20)
            plt.ylabel('Precision', fontsize=20)
            plt.title('Precision-Recall Curves', fontsize=24)
            plt.legend(lines, legends, prop=dict(size=16))

            mAP = np.mean(aps)
            print('Mean AP is %.4f' % (mAP,))
            fig_path = os.path.join(self.logdir, 'ap.png')
            plt.savefig(fig_path)
            print('Save mAPs figure in %s' % (fig_path,))
            plt.clf()

            if self.logdir:
                torch.save(running_metrics.confusion_matrix, os.path.join(self.logdir, 'conf_matrix.pth'))
                print('Save confusion matrix to %s' % (self.logdir))
                torch.save([ap_gts, ap_scores], os.path.join(self.logdir, 'gts_scores.pth'))
                print('Save gt scores to %s' % (self.logdir))

            plt.figure()
            plt.matshow(running_metrics.trimmed_conf_matrix)
            plt.title('')
            plt.colorbar()

            ax = plt.gca()
            ax.set_xticklabels([''] + cat_names)
            ax.set_yticklabels([''] + cat_names)

            fig_path = os.path.join(self.logdir, 'conf_matrix.png')
            plt.savefig(fig_path)
            print('Save confusion matrix figure in %s' % (fig_path,))

            print('Latex string')

            if self.args.is_seg == 1:
                print('%.3f & %.3f & %.3f & %.3f ' % (va.item(), score["Mean Acc : \t"],
                                                      score["Mean IoU : \t"], mAP))
            else:
                print('%.3f & %.3f & %.3f ' % (va.item(), score["Mean Acc : \t"], mAP))
