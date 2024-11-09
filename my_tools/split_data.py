import os
import random
import math
import os
import random
import math

def main(imgdir, labdir, setdir, val_ratio, test_ratio, img_type):
    datasets = []
    for msk in os.listdir(labdir):
        fname = msk.split('.')[0]
        datasets.append(fname)

    random.seed(1)  # 确保结果可复现
    random.shuffle(datasets)  # 打乱数据集顺序

    # 计算各集数量
    total_nums = len(datasets)
    test_num = math.ceil(test_ratio * total_nums)
    val_num = math.ceil(val_ratio * total_nums)
    train_num = total_nums - test_num - val_num

    # 抽取测试集
    test_set = datasets[:test_num]

    # 抽取验证集
    val_set = datasets[test_num:test_num + val_num]

    # 剩余的为训练集
    train_set = datasets[test_num + val_num:]

    # 确保划分后的数据集数量正确
    assert len(train_set) == train_num
    assert len(val_set) == val_num
    assert len(test_set) == test_num

    # 写入文件
    with open(os.path.join(setdir, 'train.txt'), 'w') as f_train:
        for name in train_set:
            f_train.write(name + '\n')

    with open(os.path.join(setdir, 'val.txt'), 'w') as f_val:
        for name in val_set:
            f_val.write(name + '\n')

    with open(os.path.join(setdir, 'test.txt'), 'w') as f_test:
        for name in test_set:
            f_test.write(name + '\n')

if __name__ == '__main__':
    imgdir = r'D:\py\engineering\mmsegmentation\data\VOCdevkit\SHANYAO\JPEGImages'
    labdir = r'D:\py\engineering\mmsegmentation\data\VOCdevkit\SHANYAO\SegmentationClass'
    setdir = r'D:\py\engineering\mmsegmentation\data\VOCdevkit\SHANYAO\ImageSets\Segmentation'
    val_ratio = 0.2  # 验证集划分比例
    test_ratio = 0.1  # 测试集划分比例
    img_type = ['.png', '.jpg']  # 原始图片的图片格式

    main(imgdir, labdir, setdir, val_ratio, test_ratio, img_type)
