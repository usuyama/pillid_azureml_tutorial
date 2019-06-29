import resnet
from tqdm import tqdm
from metrics import MetricsCollection, classification_accuracy
import copy
import time
import torch.optim
import torch.nn as nn
import torch
from dataset import PillDataset
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import pandas as pd

import azureml.core.run
# get hold of the current Azure-ML run
run = azureml.core.run.Run.get_context()

parser = argparse.ArgumentParser()

parser.add_argument('--data_root_dir', default="/home/naotous/pillidfast/data/pill_recognition_azureml_sample")
parser.add_argument('--img_dir', default="imgs")

parser.add_argument('--all_imgs_csv', default="folds/C48336_all.csv")
parser.add_argument('--val_imgs_csv', default="folds/C48336_5folds_3.csv")
parser.add_argument('--test_imgs_csv', default="folds/C48336_5folds_4.csv")

parser.add_argument('--network', default='resnet18')
parser.add_argument("--init_lr", type=float, default=5e-4, help='initial learning rate')
parser.add_argument("--batch_size", type=int, default=36)

args = parser.parse_args()

print(args)


def adjust_path(p):
    return os.path.join(args.data_root_dir, p)


args.all_imgs_csv = adjust_path(args.all_imgs_csv)
args.val_imgs_csv = adjust_path(args.val_imgs_csv)
args.test_imgs_csv = adjust_path(args.test_imgs_csv)

all_df = pd.read_csv(args.all_imgs_csv)
val_df = pd.read_csv(args.val_imgs_csv)
test_df = pd.read_csv(args.test_imgs_csv)

for df in [all_df, val_df, test_df]:
    df['images'] = df['images'].apply(lambda x: os.path.join(args.data_root_dir, args.img_dir, x))

train_df = all_df[~(all_df.images.isin(val_df.images) | all_df.images.isin(test_df.images))]

print(len(all_df), len(train_df), len(val_df), len(test_df))

label_encoder = LabelEncoder()
label_encoder.fit(all_df['label'])
print("Found {} total labels".format(len(label_encoder.classes_)))


image_datasets = {'train': PillDataset(train_df, label_encoder, augment=True),
                  'val': PillDataset(val_df, label_encoder, augment=False)}

dataloaders = {x: torch.utils.data.DataLoader(
    image_datasets[x], batch_size=args.batch_size,
    shuffle=(x == 'train')) for x in ['train', 'val']}


def train_model(model, criterion, optimizer, scheduler,
                device, dataloaders,
                label_encoder,
                num_epochs=100,
                earlystop_patience=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    has_waited = 0
    stop_training = False

    epoch_metrics = MetricsCollection()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            batch_metrics = MetricsCollection()
            predictions_list = []

            # Iterate over data.
            loader = dataloaders[phase]
            # tqdm disable=None for Azure ML (no progress-bar for non-tty)
            pbar = tqdm(loader, total=len(loader), desc="Epoch {} {}".format(epoch, phase), ncols=0, disable=None)
            for batch_index, batch_data in enumerate(pbar):
                inputs = batch_data['image'].to(device)
                labels = batch_data['label'].to(device)
                img_paths = batch_data['image_name']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)
                    scores = outputs.softmax(1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()

                batch_metrics.add(phase, 'loss', loss.item(), inputs.size(0))

                accuracies = classification_accuracy(outputs, labels, topk=(1, 5))
                batch_metrics.add(phase, 'top1-acc', accuracies[0].item(), inputs.size(0))
                batch_metrics.add(phase, 'top5-acc', accuracies[1].item(), inputs.size(0))

                pbar.set_postfix(**{k: "{:.5f}".format(meter.avg) for k, meter in batch_metrics[phase].items()})

            for key, meter in batch_metrics[phase].items():
                epoch_metrics.add(phase, key, meter.avg, 1)
                run.log('{}_{}'.format(phase, key), meter.avg)

            if phase == 'val':
                # monitor the val metrics
                best_epoch_index = epoch_metrics['val']['top1-acc'].best()[1]
                if best_epoch_index == epoch:
                    has_waited = 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("Saving the best model state dict")
                else:
                    if has_waited >= earlystop_patience:
                        print("** Early stop in training: {} waits **".format(has_waited))
                        stop_training = True

                    has_waited += 1

                if type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(epoch_metrics['val']['loss'].value)
                else:
                    scheduler.step()

        print()  # end of epoch
        if stop_training:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    best_acc, best_epoch = epoch_metrics['val']['top1-acc'].best()

    best_metrics = {
        'top1-acc': best_acc,
        'top5-acc': epoch_metrics['val']['top5-acc'].history[best_epoch],
    }

    for key, meter in epoch_metrics['val'].items():
        best_value, best_epoch = meter.best()
        train_value = epoch_metrics['train'][key].history[best_epoch]
        print('* Best val-{} at epoch {}: {:4f} (train-{}: {:4f}) *'.format(key, best_epoch, best_value, key, train_value))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_metrics


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

model = resnet.BasicResNet(num_classes=len(label_encoder.classes_), network='resnet18')
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.init_lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=1, verbose=True)

model, best_val_metrics = train_model(model, criterion, opt, scheduler, device, dataloaders, label_encoder)

# Note: file saved in the outputs folder is automatically uploaded into experiment record
# Also, you can write any files to the blob, where you have access for the input data
os.makedirs('outputs', exist_ok=True)
