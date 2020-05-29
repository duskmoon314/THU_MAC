import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
import copy
from PIL import Image
import json
# IF need to debug when using cuda
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse

parser = argparse.ArgumentParser(description="THU_MAC exp2 py")
parser.add_argument('--train_data', type=str,
                    default='./image_exp/Classification/Data/Train', help="train data dir default='./image_exp/Classification/Data/Train'")
parser.add_argument('--test_data', type=str,
                    default='./image_exp/Classification/Data/Test', help="test data dir default='./image_exp/Classification/Data/Test'")
parser.add_argument('--pretrain_weight', type=str,
                    default='./inception_v3_google-1a9a5a14.pth', help="pretrain inception weight dir default='./inception_v3_google-1a9a5a14.pth'")
parser.add_argument('--batch_size', type=int,
                    default=4, help="batch size default=4")
parser.add_argument('--lr', type=float,
                    default=0.001, help="learning rate default=0.001")
parser.add_argument('--momentum', type=float,
                    default=0.9, help="momentum default=0.9")
parser.add_argument('--step', type=int,
                    default=7, help="step size default=7")
parser.add_argument('--gamma', type=float,
                    default=0.1, help="gamma default=0.1")
parser.add_argument('--epoch', type=int,
                    default=25, help="epoch default=25")
parser.add_argument('--save_model', type=str,
                    default="./exp2_ft_model.pth", help="model save dir default='./exp2_ft_model.pth'")
parser.add_argument('--test_pred', type=str,
                    default="./exp2_pred.json", help="test pred json default='./exp2_pred.json'")
parser.add_argument('--device', type=str,
                    default="0", help="cuda device default='0'")
args = parser.parse_args()

data_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.44087456, 0.39025736, 0.43862119], [
                         0.18242574, 0.19140723, 0.18536106])
])

data_dir = args.train_data
datasets = datasets.ImageFolder(data_dir, data_transforms)
print("datasets len: {}", len(datasets))
train_db, val_db = torch.utils.data.random_split(
    datasets, [12000, (len(datasets) - 12000)])
image_datasets = {'train': train_db, 'val': val_db}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = datasets.classes

device = torch.device(
    "cuda:" + args.device if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            time_epoch = time.time() - since
            print('Training in {:.0f}m {:.0f}s'.format(
                time_epoch // 60, time_epoch % 60))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


model_ft = models.inception_v3(pretrained=False)
pre = torch.load(args.pretrain_weight)
model_ft.load_state_dict(pre)
model_ft.aux_logits = False
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(),
                         lr=args.lr, momentum=args.momentum)

exp_lr_scheduler = lr_scheduler.StepLR(
    optimizer_ft, step_size=args.step, gamma=args.gamma)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args.epoch)

exp2_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

exp2_normalize = transforms.Normalize([0.44087456, 0.39025736, 0.43862119], [
    0.18242574, 0.19140723, 0.18536106])

exp2_test_dir = args.test_data
exp2_test_data = os.listdir(exp2_test_dir)
exp2_test_result = {}
exp2_label = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5',
              'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']


def test_exp2(model, _transforms, root_dir, data_list, result):
    model.eval()
    for data_dir in data_list:
        img = Image.open(os.path.join(root_dir, data_dir))
        inputs = _transforms(img)
        inputs = exp2_normalize(inputs[0:3]).unsqueeze(0)
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        result[data_dir] = exp2_label[preds[0]]
    with open(args.test_pred, "w", newline="", encoding="utf8") as target:
        json.dump(result, target)


test_exp2(model_ft, exp2_transforms, exp2_test_dir,
          exp2_test_data, exp2_test_result)

torch.save(model_ft, args.save_model)
