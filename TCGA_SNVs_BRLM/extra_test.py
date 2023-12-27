import datetime
import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

import ResNet

torch.set_default_tensor_type(torch.DoubleTensor)


# Logger Function
def get_logger(filename, verbosity=1, name=None):
    level_dict = {
        0: logging.DEBUG,
        1: logging.INFO,
        2: logging.WARNING,
        3: logging.ERROR,
        4: logging.CRITICAL,
    }
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    # debug information filtering
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        # Convert 768 attributes to tensor
        self.data = data.iloc[:, 2:].values.astype(float)
        self.data = torch.from_numpy(self.data)
        # Converts the category label to an integer
        self.labels = data.iloc[:, 1]
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.data)  # Sample quantity obtained

    def __getitem__(self, index):
        sample = self.data[index]
        labels = self.labels[index]
        return sample, labels


# 设置随机种子及其他超参并指定训练显卡
batch_size = 32
seed = 37
device = 3
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
if torch.cuda.is_available():
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# 设置日志路径
log_path = "{}/extra_log.txt".format("log")
logger = get_logger(log_path)

# 读取数据
file_path = "data/GBM_entity_vectors.csv"
data = pd.read_csv(file_path, header=None)

# 划分数据集
train_data, test_data = train_test_split(data, train_size=0.7, random_state=seed)
extra_class3_data = test_data[test_data.iloc[:, 1] != "Class3"]
class3_data = test_data[test_data.iloc[:, 1] == "Class3"]
selected_class3_data = class3_data.sample(
    n=int(len(extra_class3_data) / 4), random_state=seed
)
test_data = pd.concat([extra_class3_data, selected_class3_data])

# 数据预处理
scaler = StandardScaler()
scaler.fit(train_data.iloc[:, 2:])
test_data_normalized = scaler.transform(test_data.iloc[:, 2:])
test_data.iloc[:, 2:] = test_data_normalized

# 创建训练集和测试集的 Dataset
test_dataset = MyDataset(test_data)

# 创建训练集和测试集的 DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 加载resnet模型
model = ResNet.resnet50(pretrained=False)
model = model.cuda(device)

# 循环加载pth文件进行测试
pth_files = sorted(glob.glob(os.path.join("checkpoints/", "*.pth")))
logger.info(pth_files)
results = []
for pth_file in pth_files:
    model.load_state_dict(torch.load(pth_file))
    # 开始训练
    step_per_epoch = len(test_dataloader)
    logger.info("#########################开始测试！###########################")
    logger.info("model:{}".format(pth_file))
    model.eval()
    test_acc = []
    for step, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.cuda(device)
        labels = labels.cuda(device)
        outputs = model(inputs)
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        if (step + 1) % 20 == 0:
            logger.info("{}/{}".format(step + 1, step_per_epoch))
            logger.info(predictions)
            logger.info(labels)
            logger.info(acc.item())
        # 记录日志
        test_acc.append(acc.item())
    test_acc = np.array(test_acc).mean()
    results.append(test_acc)
    logger.info(
        "{:%Y-%m-%d_%H:%M:%S} || model:{} || test_acc={:.6f}".format(
            datetime.datetime.now(),
            pth_file,
            test_acc,
        )
    )
    logger.info("#########################测试完成！###########################")
for i in range(len(results)):
    logger.info("{}:{}\n".format(pth_files[i], results[i]))
