import os
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from loader.init_val import name2id
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle

def get_precision_recall(labels, scores):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()

    n_classes = scores.shape[1]

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(labels[:,i], scores[:, i])
        average_precision[i] = average_precision_score(labels[:,i], scores[:, i])

    return precision, recall, average_precision




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--model_dir', type=str, default='', help='path of models')
    args = parser.parse_args()

    split_label = [0, 3, 5, 8, 9, 10]
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    id2name = {}
    for k, v in name2id.items():
        id2name[v] = k
    print(id2name)
    cat_names = [id2name[a] for a in split_label if a != 0]
    print(cat_names)

    if 1==2:
        conf_matrix = torch.load(os.path.join(args.model_dir, 'conf_matrix.pth'))
        total_pred = conf_matrix.sum(axis=1)

        trimmed_conf_matrix = []





        for i in range(11):
            if i == 0 or i not in split_label:
                continue
            arr = []
            for j in range(11):
                if j == 0 or j not in split_label:
                    continue
                if total_pred[i] == 0:
                    rt = 0
                else:
                    rt = conf_matrix[i, j] * 1.0 / total_pred[i]
                arr.append(rt)
                if rt < 0.01:
                    rt = 0
                    print('& %d   ' % (rt,), end='')
                else:
                    print('& %.2f ' % (rt,), end='')
            print()
            trimmed_conf_matrix.append(arr)
        print()


        plt.figure()
        plt.matshow(trimmed_conf_matrix)
        plt.title('')
        plt.colorbar()
        fig_path = os.path.join(args.model_dir, 'conf_matrix.png')


        ax = plt.gca()
        ax.set_xticklabels(['']+cat_names)
        ax.set_yticklabels(['']+cat_names)

        plt.savefig(fig_path)
        print('Save figure in %s' % (fig_path,))

    else:
        ap_gts, ap_scores = torch.load(os.path.join(args.model_dir, 'gts_scores.pth'))

        split_label = split_label[1:]  # trim 0


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

        print('Mean AP is %.4f' % (np.mean(aps)))
        fig_path = os.path.join(args.model_dir, 'ap.png')
        plt.savefig(fig_path)
        print('Save mAPs figure in %s' % (fig_path,))
        plt.clf()