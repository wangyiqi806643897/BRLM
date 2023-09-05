import datetime
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

import ResNet

torch.set_default_tensor_type(torch.DoubleTensor)


# Log Function
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
    # debug info filter out
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


# Custom dataset class define
class MyDataset(Dataset):
    def __init__(self, data):
        # 768-Diomentinal vectors to tensor
        self.data = data.iloc[:, 2:].values.astype(float)
        self.data = torch.from_numpy(self.data)
        # Converts category label to integer
        self.labels = data.iloc[:, 1]
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        # Converts integer labels to one-hot vectors
        # self.num_classes = len(label_encoder.classes_)
        # self.label_one_hot = torch.nn.functional.one_hot(
        #     torch.tensor(self.labels), num_classes=self.num_classes
        # )

    def __len__(self):
        return len(self.data)  # Sample quantity

    def __getitem__(self, index):
        sample = self.data[index]
        labels = self.labels[index]
        return sample, labels


# hyperparameters
train_ratio = 0.7
batch_size = 16
max_epoch = 100
weight = [1, 1, 0.03, 1, 1]
lr = 0.001
weight_decay = 0.001
patience = 8
min_lr = 0.000001
# random seeds
seed = 37
# GPU
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

# reading data
file_path = "res_CCM_Classify/entity_vectors.csv"
data = pd.read_csv(file_path, header=None)

# dataset partion
train_data, test_data = train_test_split(
    data, train_size=train_ratio, random_state=seed
)

# data preprocessing
scaler = StandardScaler()
scaler.fit(train_data.iloc[:, 2:])
train_data_normalized = scaler.transform(train_data.iloc[:, 2:])
test_data_normalized = scaler.transform(test_data.iloc[:, 2:])
train_data.iloc[:, 2:] = train_data_normalized
test_data.iloc[:, 2:] = test_data_normalized

# training and test Dataset
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)

# training and testing DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# resnet model
model = ResNet.resnet50(pretrained=False)

# training on GPU
model = model.cuda(device)

# Loss Function and Optimizer definition
loss_func = nn.CrossEntropyLoss(weight=torch.tensor(weight).cuda(device))
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    patience=patience,
    min_lr=min_lr,
)

# tensorboard and logger
sw = SummaryWriter("log")
log_path = "{}/log.txt".format("log")
logger = get_logger(log_path)

# Begin Training
global_step = 0
step_per_epoch = len(train_dataloader)
step_per_epoch2 = len(test_dataloader)
for epoch in range(max_epoch):
    logger.info("######################### Training Begin！###########################")
    model.train()
    loss_per_ten_step = 0
    acc_per_ten_step = 0
    show_step = 30
    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.cuda(device)
        labels = labels.cuda(device)
        # Gradient cumulative clearing
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        if (step + 1) % 30 == 0:
            logger.info("{}/{}".format(step + 1, step_per_epoch))
            logger.info(predictions)
            logger.info(labels)
            logger.info(acc.item())
        # log record
        global_step += 1
        loss_per_ten_step += loss.item()
        acc_per_ten_step += acc.item()
        if ((step + 1) % show_step == 0) or ((step + 1) == step_per_epoch):
            if (step + 1) % show_step == 0:
                loss_per_ten_step = loss_per_ten_step / show_step
                acc_per_ten_step = acc_per_ten_step / show_step
            else:
                leave_steps = step_per_epoch % show_step
                loss_per_ten_step = loss_per_ten_step / leave_steps
                acc_per_ten_step = acc_per_ten_step / leave_steps
            sw.add_scalar(
                "lr",
                optimizer.param_groups[0]["lr"],
                epoch + 1,
            )
            sw.add_scalar("loss", loss_per_ten_step, global_step=global_step)
            sw.add_scalar("acc", acc_per_ten_step, global_step=global_step)
            logger.info(
                "{:%Y-%m-%d_%H:%M:%S} || step:{:.0f}/{:.0f} || epoch:{:.0f}/{:.0f} || lr={:.7f} || loss={:.6f} || acc={:.6f}".format(
                    datetime.datetime.now(),
                    step + 1,
                    step_per_epoch,
                    epoch + 1,
                    max_epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_per_ten_step,
                    acc_per_ten_step,
                )
            )
            loss_per_ten_step = 0
            acc_per_ten_step = 0
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), "checkpoints/model-{}.pth".format(epoch + 1))
    logger.info("######################### Training Finish！###########################")
    logger.info("######################### Testing Begin！###########################")
    model.eval()
    test_loss = []
    test_acc = []
    prediction_list=[]
    show_step = 30
    for step, (inputs, labels) in enumerate(test_dataloader):
        inputs = inputs.cuda(device)
        labels = labels.cuda(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        ret, predictions = torch.max(outputs.data, 1)
        pl = predictions.tolist()
        for i in range(0,len(pl)):
            prediction_list.append(Class[pl[i]])
        logger.info("{}/{}".format(step + 1, step_per_epoch2))
        logger.info(predictions)
        logger.info(labels)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        logger.info(acc.item())
        # log record
        test_loss.append(loss.item())
        test_acc.append(acc.item())
    test_loss = np.array(test_loss).mean()
    test_acc = np.array(test_acc).mean()
    logger.info(
        "{:%Y-%m-%d_%H:%M:%S} || epoch:{:.0f}/{:.0f} || test_loss={:.6f} || test_acc={:.6f}".format(
            datetime.datetime.now(),
            epoch + 1,
            max_epoch,
            test_loss,
            test_acc,
        )
    )
    sw.add_scalar("test_loss", test_loss, global_step=epoch + 1)
    sw.add_scalar("test_acc", test_acc, global_step=epoch + 1)
    scheduler.step(test_loss)
    prediction_res=open("res_CCM_Classify/Prediction_Class.txt","w")
    variant_list = list(test_data.iloc[:, 0])
    for i in range(0,len(test_data)):
        prediction_res.write(variant_list[i]+"\t"+prediction_list[i]+"\n")
    prediction_res.close()
    logger.info("######################### Testing Finish！###########################")
