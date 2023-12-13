import os
import sys
import time

import hydra
from omegaconf import OmegaConf

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from utils.logger import Logger, Recorder
from utils.dataset import create_dataset
from utils.function import reshape_image, visualize_record, summarize_record
from model.transformer import SpikingTransformer


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    # load configuration
    checkpoint = str(config.checkpoint)
    device = torch.device('cuda') if str(config.device) == 'gpu' else torch.device('cpu')

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint, 'training.log'))
    config.checkpoint = str(checkpoint)
    config.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    model = SpikingTransformer(in_channels=config.dataset.in_channels, image_size=config.dataset.image_size, num_classes=config.dataset.num_classes, num_layers=config.model.num_layers, num_heads=config.model.num_heads, num_channels=config.model.num_channels).to(device)
    functional.set_step_mode(model, 'm')

    # create optimizer
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.save_interval)

    # load dataset
    dataset_train, dataset_test = create_dataset(config.dataset)
    loader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    loader_test = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    print('train dataset: {} samples, test dataset: {} samples\n'.format(len(dataset_train), len(dataset_test)))

    # start training
    recorder = Recorder('epoch', 'loss_train', 'accuracy_train', 'time_train', 'loss_test', 'accuracy_test', 'time_test')
    print('num_epochs: {}\n'.format(config.num_epochs))

    for epoch in range(1, config.num_epochs + 1):
        print('epoch start: {} / {}'.format(epoch, config.num_epochs))

        # train
        loss_train = 0
        correct_train = 0
        start_train = time.time()
        model.train()

        for image, label in loader_train:
            image, label = image.to(device), label.to(device)
            image = reshape_image(image, config.dataset.time_steps)

            optimizer.zero_grad()
            prediction = model(image)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            correct_train += (prediction.argmax(dim=1) == label).sum().item()
            functional.reset_net(model)

        scheduler.step()
        loss_train /= len(dataset_train)
        accuracy_train = correct_train / len(dataset_train)
        time_train = time.time() - start_train

        # test
        loss_test = 0
        correct_test = 0
        start_test = time.time()
        model.eval()

        with torch.no_grad():
            for image, label in loader_test:
                image, label = image.to(device), label.to(device)
                image = reshape_image(image, config.dataset.time_steps)

                prediction = model(image)
                loss = criterion(prediction, label)

                loss_test += loss.item()
                correct_test += (prediction.argmax(dim=1) == label).sum().item()
                functional.reset_net(model)

        loss_test /= len(dataset_test)
        accuracy_test = correct_test / len(dataset_test)
        time_test = time.time() - start_test

        # report
        recorder.record({'epoch': epoch, 'loss_train': loss_train, 'accuracy_train': accuracy_train, 'time_train': time_train, 'loss_test': loss_test, 'accuracy_test': accuracy_test, 'time_test': time_test})
        print('epoch finish: {} / {}'.format(epoch, config.num_epochs))
        print('loss_train: {:.6f}, accuracy_train: {:.2%}, time_train: {:.2f}'.format(loss_train, accuracy_train, time_train))
        print('loss_test: {:.6f}, accuracy_test: {:.2%}, time_test: {:.2f}\n'.format(loss_test, accuracy_test, time_test))

        # save
        if epoch % config.save_interval == 0:
            save_path = os.path.join(checkpoint, 'epoch_{}'.format(epoch))
            os.makedirs(save_path, exist_ok=True)
            visualize_record(save_path, recorder.records)
            summarize_record(save_path, recorder.records)
            recorder.save(os.path.join(save_path, 'record.pkl'))
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
            print('save checkpoint: {}\n'.format(save_path))


if __name__ == '__main__':
    main()
