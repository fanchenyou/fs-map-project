# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import torch

###########################
# evaluate segmentation ###
###########################
class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_agent = 0
        self.total_bandW = 0
        self.count = 0

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_agent = 0
        self.total_bandW = 0
        self.count = 0

    def print_score(self, score, class_iou, split_label, ignore_bg=True):
        metric_string = ""
        class_string = ""

        for i in range(self.n_classes):
            # print(i, class_iou[i])
            metric_string = metric_string + "  " + str(i)
            class_string = class_string + " " + str(round(class_iou[i] * 100, 2))

        for k, v in score.items():
            metric_string = metric_string + "  " + str(k)
            class_string = class_string + " " + str(round(v * 100, 2))
            # print(k, v)
        print(metric_string)
        print(class_string)

        print('Confusion matrix')
        conf_matrix = self.confusion_matrix
        trimmed_conf_matrix = []

        total_pred = conf_matrix.sum(axis=1)

        for i in range(self.n_classes):
            if (ignore_bg and i == 0) or i not in split_label:
                continue
            arr = []
            for j in range(self.n_classes):
                if (ignore_bg and j == 0) or j not in split_label:
                    continue
                if total_pred[i] == 0:
                    rt = 0
                else:
                    rt = conf_matrix[i, j] * 1.0 / total_pred[i]
                arr.append(rt)
                if rt<0.01:
                    rt = 0
                    print('& %d    ' % (rt,), end='')
                else:
                    print('& %.2f ' % (rt,), end='')
            print()
            trimmed_conf_matrix.append(arr)
        print()

        trimmed_conf_matrix = np.array(trimmed_conf_matrix)
        self.trimmed_conf_matrix = trimmed_conf_matrix





###########################
# evaluate classification #
###########################
class averageMeter(object):
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
