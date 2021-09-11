import torch
import numpy as np


# For segmentation
class CategoriesSampler:

    def __init__(self, all_labels, label_stat, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        # sanity check the number of instances of each class
        for k, n in label_stat.items():
            assert n >= self.n_per

        self.label_stat = label_stat
        # print(self.label_stat)
        labels = np.array(sorted(list(self.label_stat.keys())))  # all data label
        print(labels)

        all_labels = np.array(all_labels)  # all data label
        self.m_ind = {}  # the data index of each class
        for i in labels:
            # print(i)
            ind = np.argwhere(all_labels == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

        self.labels = labels
        self.num_labels = len(self.labels) - 1  # discard bg class 0
        # print(self.num_labels, self.n_cls,'----')
        # assert self.num_labels >= self.n_cls
        self.fixed_batches = []
        self.fixed_batches_classes = []
        self.mode = 'rand'  # 'probe', 'fix'

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        if self.mode == 'rand':
            for i_batch in range(self.n_batch):
                batch = []
                classes = torch.randperm(self.num_labels)[:self.n_cls]  # random sample num_class indices,e.g. 5
                for c in classes:
                    cls = self.labels[c + 1]
                    l = self.m_ind[cls]  # all data indexs of this class
                    pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                    batch.append(l[pos])
                batch = torch.stack(batch).t().reshape(-1)
                # .t() transpose,
                # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
                # instead of aaaabbbbccccdddd
                yield batch

        elif self.mode == 'probe':
            for i_batch in range(self.n_batch):
                batch = []
                classes = torch.randperm(self.num_labels)[:self.n_cls]  # random sample num_class indices,e.g. 5
                for c in classes:
                    cls = self.labels[c + 1]
                    l = self.m_ind[cls]  # all data indexs of this class
                    pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                    batch.append(l[pos])

                self.fixed_batches.append(batch)
                batch_t = torch.stack(batch).t().reshape(-1)
                # .t() transpose,
                # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
                # instead of aaaabbbbccccdddd
                yield batch_t

        else:
            assert self.mode == 'fix'
            assert len(self.fixed_batches)==self.n_batch
            #print(self.fixed_batches)
            for ix, batch in enumerate(self.fixed_batches):
                batch_t = torch.stack(batch).t().reshape(-1)
                yield batch_t


# For segmentation
# not deterministic for validation
class CategoriesSampler_v11:

    def __init__(self, all_labels, label_stat, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        # sanity check the number of instances of each class
        for k, n in label_stat.items():
            if n < self.n_per:
                del label_stat[k]

        self.label_stat = label_stat
        # print(self.label_stat)
        labels = np.array(sorted(list(self.label_stat.keys())))  # all data label
        # print(labels)

        all_labels = np.array(all_labels)  # all data label
        self.m_ind = {}  # the data index of each class
        for i in labels:
            # print(i)
            ind = np.argwhere(all_labels == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind[i] = ind

        self.labels = labels
        self.num_labels = len(self.labels) - 1  # discard bg class 0
        # print(self.num_labels, self.n_cls,'----')
        # assert self.num_labels >= self.n_cls

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(self.num_labels)[:self.n_cls]  # random sample num_class indices,e.g. 5
            for c in classes:
                cls = self.labels[c + 1]
                l = self.m_ind[cls]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch


class CategoriesClassificationSampler:

    def __init__(self, all_label, split_label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per
        self.split_label = split_label

        label = np.array(all_label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indices,e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
