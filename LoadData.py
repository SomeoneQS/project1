import torch.utils.data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from ProcessData import processData_Col
import numpy as np
import gc
import pandas as pd
import math
import platform
from ProcessData.DataSet_IDRID import DataSetIDRIR


class loadData():
    def __init__(self, root_train, root_test, batchsize, root_data='./Data/images', verbose=print):
        self.ROOT_TRAIN = root_train
        self.ROOT_TEST = root_test
        self.BatchSize = batchsize
        self.train_set = None
        self.test_set = None
        self.data_set = None
        self.load_data = False
        self.ROOT_DATA = root_data
        self.classes = None
        self.verbose = verbose

    def get_full_data_without_test(self, val_per=0.2, sample=False, sample_val=False):
        from ProcessData.processData_Col import g_full_csv, get_Mean, get_Std
        mean = processData_Col.full_mean
        std = processData_Col.full_std
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.3),  # 随机垂直翻转
            transforms.RandomHorizontalFlip(p=0.6),  # 随机水平翻转
            transforms.ToTensor(),
            normalize
        ])
        trans_val = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        data_set = ImageFolder('./Data/images', transform=transform)
        self.load_data = True
        train_set, val_set = random_split(data_set, [math.ceil(len(data_set) * (1 - val_per)),
                                                     math.floor(len(data_set) * val_per)])
        val_set.dataset.transform = trans_val
        sampler, weights = self.get_Weighted_sampler(train_set)
        if sample:
            train_loader = DataLoader(train_set, batch_size=self.BatchSize, drop_last=True, sampler=sampler)
        else:
            train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        if sample_val and sample:
            sampler_val, _ = self.get_Weighted_sampler(val_set)
            val_loader = DataLoader(val_set, batch_size=self.BatchSize, drop_last=True, sampler=sampler_val)
        else:
            val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        # weights = torch.softmax(torch.tensor(train_weight), dim=-1).cpu().numpy()
        return train_loader, val_loader, None, weights

    def _return_data_(self):
        if self.load_data:
            return
        normalize = transforms.Normalize(mean=processData_Col.data_mean, std=processData_Col.data_std)
        transform = transforms.Compose([
            # transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
            # transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
            transforms.ToTensor(),
            normalize
        ])
        self.data_set = ImageFolder(self.ROOT_DATA, transform=transform)
        self.load_data = True

    def _return_data_set_(self, do_k=False):
        if self.load_data:
            return
        mean, std = self.getStat(
            DataLoader(ImageFolder(self.ROOT_TRAIN, transform=transforms.ToTensor()), batch_size=self.BatchSize))
        normalize_train = transforms.Normalize(mean=mean, std=std)
        # normalize_test = transforms.Normalize(mean=processData_Col.mean_test, std=processData_Col.std_test)
        train_transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # 裁剪为256 * 256，数据集图像只有 224 * 224
            transforms.RandomVerticalFlip(p=0.3),  # 随机垂直翻转
            transforms.RandomHorizontalFlip(p=0.6),  # 随机水平翻转
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            normalize_train])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize_train])
        train_dataset = ImageFolder(self.ROOT_TRAIN, transform=train_transform)
        test_dataset = ImageFolder(self.ROOT_TEST, transform=test_transform)
        self.train_set = train_dataset
        self.test_set = test_dataset
        self.load_data = True
        if do_k:
            self.data_set = ConcatDataset([self.train_set, self.test_set])
            # 回收内存
            del self.train_set
            del self.test_set
            gc.collect()

    def get_data_loader(self):
        self._return_data_set_()
        # train_loader = [(data, label), ..., (data, label)]
        train_loader = DataLoader(self.train_set, batch_size=self.BatchSize, shuffle=True, drop_last=False)
        test_loader = DataLoader(self.test_set, batch_size=self.BatchSize, shuffle=True, drop_last=False)
        _, weights = self.get_Weighted_sampler(self.train_set)
        # weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().numpy()
        return train_loader, test_loader, None, weights

    def get_data_loader_val(self):
        self._return_data_set_()
        # 从验证集中划分测试集，验证不在val集中使用数据增强的影响
        train_loader = DataLoader(self.train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        val_set, test_set = random_split(self.test_set,
                                         [int(len(self.test_set) * 0.4), math.ceil(len(self.test_set) * 0.6)])
        val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        _, weights = self.get_Weighted_sampler(self.train_set)
        # weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().numpy()
        return train_loader, val_loader, test_loader, weights

    def get_imgesClahe(self, train_per=0.8, val_per=0.1, weight_softmax=False):
        if val_per < 0.0 or val_per > 1.0:
            return None
        if train_per < 0.0 or train_per > 1.0:
            return None
        if train_per + val_per > 1.0:
            return None
        IMG_ROOT = "./Data/images_clahe"
        Data_SET_G = ImageFolder(IMG_ROOT, transform=transforms.ToTensor())
        val_num = int(len(Data_SET_G) * val_per)
        train_num = int(len(Data_SET_G) * train_per)
        split_list = [train_num, val_num, len(Data_SET_G) - val_num - train_num]
        train_set, val_set, test_set = random_split(Data_SET_G, split_list)
        print("Counting mean std...")
        mean, std = self.getStat(DataLoader(train_set, batch_size=self.BatchSize, shuffle=False, drop_last=False))
        print("Counting weight...")
        sampler, weights = self.get_Weighted_sampler(train_set)
        # if weight_softmax:
        #     weights = torch.softmax(torch.tensor(train_weight), dim=-1).cpu().clone().detach().numpy()
        # else:
        #     weights = train_weight.cpu().clone().detach().numpy()
        normlize = transforms.Normalize(mean=mean, std=std)
        trans_train = transforms.Compose([
            # transforms.Resize((224, 224)),  # 裁剪为256 * 256，数据集图像只有 224 * 224
            # transforms.RandomVerticalFlip(p=0.3),  # 随机垂直翻转
            # transforms.RandomHorizontalFlip(p=0.6),  # 随机水平翻转
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            normlize
        ])
        # chage transforms
        Data_SET_G.transforms = trans_train
        train_loader = DataLoader(train_set, batch_size=self.BatchSize, drop_last=False, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.BatchSize, drop_last=False, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.BatchSize, drop_last=False, shuffle=False)
        return train_loader, val_loader, test_loader, weights

    def get_data_with_val(self, train_per=0.8, val_per=0.1):
        self._return_data_()
        if val_per < 0.0 or val_per > 1.0:
            return None
        if train_per < 0.0 or train_per > 1.0:
            return None
        if train_per + val_per > 1.0:
            return None
        # 先划分数据集,再在各个子数据集上进行 WeightedRandomSampler
        val_num = int(len(self.data_set) * val_per)
        train_num = int(len(self.data_set) * train_per)
        split_list = [train_num, val_num, len(self.data_set) - val_num - train_num]
        train_set, val_set, test_set = random_split(self.data_set, split_list)
        mean, std = self.getStat(DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True))
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            # transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
            # transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
            transforms.ToTensor(),
            normalize
        ])
        self.data_set.transform = transform
        _, weights = self.get_Weighted_sampler(train_set)
        train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        # weights = torch.softmax(torch.tensor(train_weight), dim=-1).cpu().numpy()
        return train_loader, val_loader, test_loader, weights

    def get_Weighted_sampler(self, dataset):
        classes = [label for _, label in dataset]
        class_count = [0] * len(set(classes))
        for i in classes:
            class_count[i] += 1
        print(f"0/1/2/3/4: {class_count}")
        weights = len(classes) / (len(class_count) * np.array(class_count))
        weights = np.array(weights).astype(dtype='float32')
        # weights是每个类别在dataset样本中的占比，weights[classes]即把classes中的每个样本转换为对应的权重，
        # 因为weights按顺序排列，所以classes中的一个样本，如4，则其对应的权重为weights[4]
        samples_weights = weights[classes]
        return WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights),
                                     replacement=True), weights

    def get_data_with_val_in_balance(self, train_per=0.8, val_per=0.1, sample_val=False):
        self._return_data_()
        if val_per < 0.0 or val_per > 1.0:
            return None
        if train_per < 0.0 or train_per > 1.0:
            return None
        if train_per + val_per > 1.0:
            return None
        # 先划分数据集,再在各个子数据集上进行 WeightedRandomSampler
        val_num = int(len(self.data_set) * val_per)
        train_num = int(len(self.data_set) * train_per)
        split_list = [train_num, val_num, len(self.data_set) - val_num - train_num]
        train_set, val_set, test_set = random_split(self.data_set, split_list)
        sampler, weights = self.get_Weighted_sampler(train_set)
        train_loader = DataLoader(train_set, batch_size=self.BatchSize, drop_last=True, sampler=sampler)
        if sample_val:
            sampler_val, _ = self.get_Weighted_sampler(val_set)
            val_loader = DataLoader(val_set, batch_size=self.BatchSize, drop_last=True, sampler=sampler_val)
        else:
            val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=self.BatchSize, shuffle=False, drop_last=True)
        # weights = torch.softmax(torch.tensor(train_weight), dim=-1).cpu().numpy()
        return train_loader, val_loader, test_loader, weights

    def get_data_K_fold_yiled(self, k):
        self._return_data_set_(do_k=True)
        t_size = len(self.data_set)
        fold_num = int(t_size / k)
        split_list = [fold_num for i in range(k - 1)]
        split_list.append(t_size - (k - 1) * fold_num)
        # 固定种子，让每把划分都一样
        folds = random_split(self.data_set, split_list, generator=torch.manual_seed(0))
        for i in range(k):
            val_set = folds[i]
            train_sets = [x for j, x in enumerate(folds) if j != i]
            train_set = ConcatDataset(train_sets)
            train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
            yield train_loader, val_loader
            del train_loader, train_set, train_sets
            del val_set, val_loader
            gc.collect()

    def get_data_K_fold_data_set(self, k):
        self._return_data_()
        t_size = len(self.data_set)
        fold_num = int(t_size / k)
        split_list = [fold_num for i in range(k - 1)]
        split_list.append(t_size - (k - 1) * fold_num)
        # 固定种子，让每把划分都一样
        folds = random_split(self.data_set, split_list)
        return folds

    def get_data_K_fold_data_mes(self, k):
        if platform.system().lower() == "windows":
            image_root = './Data/images'
        else:
            image_root = './Data/mesdata/images'

    def get_mes_data(self, is_softmax: bool = True, processed: bool = True, auto_split: bool = False):
        # if auto_split:
        #     from ProcessData.processMes import do_split_data
        #     print("Processing data...")
        #     if processed:
        #         do_split_data((0.7, 0.1, 0.2), is_pre=True)
        #     else:
        #         do_split_data((0.7, 0.1, 0.2), is_pre=False)
        print("Counting mean and std...")
        mean, std = self.getStat(DataLoader(ImageFolder(r'./Data/mesdata/train', transform=transforms.ToTensor()),
                                            batch_size=self.BatchSize, shuffle=False, drop_last=False))
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normlize = transforms.Normalize(mean=mean, std=std)
        trans_train = transforms.Compose([
            # transforms.Resize((224, 224)),  # 裁剪为256 * 256，数据集图像只有 224 * 224
            transforms.RandomVerticalFlip(p=0.7),  # 随机垂直翻转
            transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            normlize
        ])
        trans_val = transforms.Compose([
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            # transforms.Normalize(mean=processMes.val_mean, std=processMes.val_std)
            normlize
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            # transforms.Normalize(mean=processMes.test_mean, std=processMes.test_std)
            normlize
        ])
        train_loader = DataLoader(ImageFolder(r'./Data/mesdata/train', transform=trans_train),
                                  batch_size=self.BatchSize, shuffle=True, drop_last=False)
        val_loader = DataLoader(ImageFolder(r'./Data/mesdata/val', transform=trans_val), batch_size=self.BatchSize,
                                shuffle=True, drop_last=False)
        test_loader = DataLoader(ImageFolder(r'./Data/mesdata/test', transform=trans_test), batch_size=self.BatchSize,
                                 shuffle=True, drop_last=False)
        _, weights = self.get_Weighted_sampler(ImageFolder(r'./Data/mesdata/train', transform=trans_train))
        # if is_softmax:
        #     weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().numpy()
        return train_loader, val_loader, test_loader, weights



    def get_mes_data_from_all(self, train_per_total=0.8, train_per_val=0.9):
        if platform.system().lower() == "windows":
            image_root = './Data/images'
        else:
            image_root = './Data/mesdata/images'

    def getStat(self, data_loader):
        '''
        Compute mean and variance for training data
        :param train_data: 自定义类Dataset(或ImageFolder即可)
        :return: (mean, std)
        '''
        # print('Compute mean and variance for training data.')
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for idx, (X, _) in enumerate(data_loader):
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
            print(f"\r<{idx + 1}>[{(idx + 1) * len(X)}/{len(data_loader.dataset)}]正在计算数据集信息...", end='')
        print()
        mean.div_(len(data_loader))
        std.div_(len(data_loader))
        self.verbose(f"mean: {mean}, std: {std}")
        return list(mean.numpy()), list(std.numpy())

    def get_IDRIR_set(self, is_test=True, size=256, val_size=0.1):
        if platform.system().lower() == "windows":
            image_roots = (r"D:\Academic\DataSet\datasets\IDRID\B. Disease Grading\1. Original Images\a. Training Set",
                           r"D:\Academic\DataSet\datasets\IDRID\B. Disease Grading\1. Original Images\b. Testing Set")
            labels = (
                r"D:\Academic\DataSet\datasets\IDRID\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv",
                r"D:\Academic\DataSet\datasets\IDRID\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv")
        else:
            image_roots = (r"./Data/IDRID/B. Disease Grading/1. Original Images/a. Training Set",
                           r"./Data/IDRID/B. Disease Grading/1. Original Images/b. Testing Set")
            labels = (
                r"./Data/IDRID/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
                r"./Data/IDRID/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

        tmp = transforms.Compose([
            transforms.ToTensor(),
        ])
        # 加载训练集
        trainSet = DataSetIDRIR(image_roots[0], labels[0], transform=tmp, data_type='train')
        # 计算目标值
        train_loader = DataLoader(trainSet, batch_size=self.BatchSize, shuffle=True, drop_last=False)
        mean, std = self.getStat(train_loader)
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            # transforms.Normalize(mean=processMes.val_mean, std=processMes.val_std)
            transforms.Normalize(mean=mean, std=std)
        ])
        # 修改transform
        trainSet.transform = transform
        # 划分验证集
        if val_size > 0:
            train_length = math.ceil(len(trainSet) * (1 - val_size))
            train_set, val_set = random_split(trainSet, [train_length, len(trainSet) - train_length])
            train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=False)
        else:
            train_loader = DataLoader(trainSet, batch_size=self.BatchSize, shuffle=True, drop_last=False)
        if is_test:
            test_loader = DataLoader(DataSetIDRIR(image_roots[1], labels[1], transform=tmp, data_type='test'),
                                     batch_size=self.BatchSize, shuffle=True,
                                     drop_last=False)
        _, weights = self.get_Weighted_sampler(train_set)
        # weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().numpy()
        if val_size == 0.:
            return train_loader, test_loader, None, weights
        return train_loader, val_loader, test_loader, weights

    def get_mes_1_tvt(self, train_pre=0.8, val_pre=0.2, size=256, weighted=False, processed=True, weight_softmax=True):
        # if platform.system().lower() == "windows":
        #     image_root = 'D:/Academic/DataSet/datasets/Messidor/Messidor1'
        # else:
        #     image_root = './Data/Messidor1'
        # if processed:
        #     image_root = image_root.replace("Messidor1", "p_Messidor1")
        # tmp = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        # data_set = ImageFolder(image_root, transform=tmp)
        # data_len = len(data_set)
        # train_len = math.ceil(data_len * train_pre)
        # # val from train
        # val_len = int(train_len * val_pre)
        # train_set, test_set = random_split(data_set, [train_len, data_len - train_len])
        # train_set, val_set = random_split(train_set, [train_len - val_len, val_len])
        # train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=False)
        # mean, std = self.getStat(train_loader)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
        #     transforms.Normalize(mean=mean, std=std)
        # ])
        # sampler, weights = self.get_Weighted_sampler(train_set)
        # if weighted:
        #     train_loader = DataLoader(train_set, batch_size=self.BatchSize, drop_last=True, sampler=sampler)
        # else:
        #     train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        # if weight_softmax:
        #     weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().numpy()
        # data_set.transform = transform
        # # train_loader = DataLoader(train_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        # val_loader = DataLoader(val_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        # test_loader = DataLoader(test_set, batch_size=self.BatchSize, shuffle=True, drop_last=True)
        # return train_loader, val_loader, test_loader, weights
        # ***************************
        # ***************************
        train_root = "./Data/m1_train"
        val_root = "./Data/m1_val"
        test_root = "./Data/m1_test"
        mean, std = self.getStat(DataLoader(ImageFolder(train_root, transform=transforms.ToTensor())))
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        sampler, weights = self.get_Weighted_sampler(ImageFolder(train_root, transform=transforms.ToTensor()))
        train_loader = DataLoader(ImageFolder(train_root, transform=train_transform),
                                  batch_size=self.BatchSize, shuffle=True, drop_last=False)
        val_loader = DataLoader(ImageFolder(val_root, transform=transform),
                                batch_size=self.BatchSize, shuffle=True, drop_last=False)
        test_loader = DataLoader(ImageFolder(test_root, transform=transform),
                                 batch_size=self.BatchSize, shuffle=True, drop_last=False)
        # if weight_softmax:
        #     weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().detach().clone().numpy()
        # else:
        #     weights = weights.cpu().detach().clone().numpy()
        return train_loader, val_loader, test_loader, weights
        # return None, None, None, None

    def get_kfold_data_Stratified(self, k, data_root="./Data/p_Messidor1", fill=0, num_classes=5):
        from torch.utils.data.dataset import Subset
        mean, std = self.getStat(DataLoader(ImageFolder(data_root, transform=transforms.ToTensor())))
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(0.7),
            transforms.RandomHorizontalFlip(0.7),
            # transforms.RandomRotation((90, 90), fill=fill),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        # 获取所有标签和索引
        dataset = ImageFolder(data_root, transform=transforms.ToTensor())
        _, weights = self.get_Weighted_sampler(dataset)
        # classes = [label for _, label in dataset]
        # class_count = [0] * len(set(classes))
        # for i in classes:
        #     class_count[i] += 1
        self.verbose("正在计算类别信息。。。")
        class_count = get_classes_count(dataset)
        each_class_num_in_fold = []
        for i in range(num_classes):
            each_class_num_in_fold.append(class_count[i] // k)
        shuffled_classes_idx = []
        start_idx = 0
        for i in range(num_classes):
            tmp = np.arange(start_idx, start_idx + class_count[i])
            start_idx += class_count[i]
            np.random.shuffle(tmp)
            shuffled_classes_idx.append(tmp.tolist())
        folders = []
        self.verbose("正在划分数据集。。。")
        for i in range(k - 1):
            idxs = []
            for j in range(num_classes):
                start_idx = i * each_class_num_in_fold[j]
                end_idx = (i + 1) * each_class_num_in_fold[j]
                idx_to_take = shuffled_classes_idx[j][start_idx:end_idx]
                idxs += idx_to_take
            folders.append(Subset(dataset, indices=idxs))
        # last fold
        idxs = []
        for j in range(num_classes):
            idx_to_take = shuffled_classes_idx[j][(k - 1) * each_class_num_in_fold[j]:]
            idxs += idx_to_take
        folders.append(Subset(dataset, indices=idxs))
        return folders, transform, transform_val, weights

    def load_fake_4(self):
        mean = [0.4961, 0.2507, 0.1049]
        std = [0.3244, 0.1645, 0.0709]
        root_path = "./Data/mesdata/imagesp"
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        dataset = ImageFolder(root_path, transform=transform_val)
        return DataLoader(dataset=dataset, batch_size=self.BatchSize, shuffle=True, drop_last=False)

    def get_apto_data(self, is_softmax: bool = True, processed: bool = True, auto_split: bool = False):
        # if auto_split:
        #     from ProcessData.processMes import do_split_data
        #     print("Processing data...")
        #     if processed:
        #         do_split_data((0.7, 0.1, 0.2), is_pre=True)
        #     else:
        #         do_split_data((0.7, 0.1, 0.2), is_pre=False)
        print("Counting mean and std...")
        mean, std = self.getStat(DataLoader(ImageFolder(r'./Data/APTO/train', transform=transforms.ToTensor()),
                                            batch_size=self.BatchSize, shuffle=False, drop_last=False))
        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normlize = transforms.Normalize(mean=mean, std=std)
        trans_train = transforms.Compose([
            # transforms.Resize((224, 224)),  # 裁剪为256 * 256，数据集图像只有 224 * 224
            transforms.RandomVerticalFlip(p=0.7),  # 随机垂直翻转
            transforms.RandomHorizontalFlip(p=0.3),  # 随机水平翻转
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            normlize
        ])
        trans_val = transforms.Compose([
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            # transforms.Normalize(mean=processMes.val_mean, std=processMes.val_std)
            normlize
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),  # 将0-255范围的像素转为0-1.0范围的tensor
            # transforms.Normalize(mean=processMes.test_mean, std=processMes.test_std)
            normlize
        ])
        train_loader = DataLoader(ImageFolder(r'./Data/APTO/train', transform=trans_train),
                                  batch_size=self.BatchSize, shuffle=True, drop_last=False)
        val_loader = DataLoader(ImageFolder(r'./Data/APTO/val', transform=trans_val), batch_size=self.BatchSize,
                                shuffle=True, drop_last=False)
        test_loader = DataLoader(ImageFolder(r'./Data/APTO/test', transform=trans_test), batch_size=self.BatchSize,
                                 shuffle=True, drop_last=False)
        _, weights = self.get_Weighted_sampler(ImageFolder(r'./Data/APTO/train', transform=trans_train))
        # if is_softmax:
        #     weights = torch.softmax(torch.tensor(weights), dim=-1).cpu().numpy()
        return train_loader, val_loader, test_loader, weights



def get_classes_count(dataset):
    classes = [label for _, label in dataset]
    class_count = [0] * len(set(classes))
    for i in classes:
        class_count[i] += 1
    return class_count




if __name__ == '__main__':
    # MEAN_RGB = [0.316, 0.222, 0.157]
    # STD_RGB = [0.301, 0.217, 0.171]
    # BATCH_SIZE = 64
    # data = loadData('./Data/train', './Data/test', MEAN_RGB, STD_RGB, 64)
    # train_loader, test_loader = data.get_data_loader()
    # for idx, (inputs, labels) in enumerate(train_loader):
    #     print(idx, inputs.shape, labels)
    #     break
    pass
