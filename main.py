import os
import sys

from omegaconf import DictConfig, open_dict
import hydra
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid

from dataset import MyDataset
from model import Attention, Additive


def save_img(X, path='./img/', filename='img', nrow=4, mean=torch.tensor([0.5]), std=torch.tensor([0.5])):
    X = make_grid(X, nrow=nrow, padding=0)[0]
    X = X * std + mean
    np.savetxt(path + filename, X.numpy(), delimiter=',')


def save_score(S, path=f'./score/', filename='score', nrow=4):
    S = S.contiguous().view(nrow, nrow)
    np.savetxt(path + filename, S.numpy(), delimiter=',')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    sys.stdout = open('stdout.txt', 'w')
    os.makedirs('img')
    os.makedirs('score')

    with open_dict(cfg):
        cfg.use_cuda = cfg.use_cuda and torch.cuda.is_available()

    torch.manual_seed(cfg.seed)
    if cfg.use_cuda:
        print(torch.cuda.get_device_name())
        torch.cuda.manual_seed(cfg.seed)

    print('Load Train and Test Set')
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyDataset(train=True, **cfg.dataset),
        batch_size=cfg.settings.batch_size,
        shuffle=True,
        **loader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        MyDataset(train=False, **cfg.dataset),
        batch_size=1,
        shuffle=False,
        **loader_kwargs)

    print('Init Model')
    if cfg.model.name == 'attention':
        model = Attention()
    elif cfg.model.name == 'additive':
        model = Additive()
    if cfg.use_cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg.settings.lr, betas=(0.9, 0.999), weight_decay=cfg.settings.reg)


    def train(epoch):
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loss = 0.
        train_accuracy = 0.

        for i, (X_batch, y_batch) in enumerate(train_loader):
            if cfg.use_cuda:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            optimizer.zero_grad()

            y_proba_list = []
            y_hat_list = []
            for j, (X, y) in enumerate(zip(X_batch, y_batch)):
                X, y = X.unsqueeze(0), y.unsqueeze(0)
                y_proba, y_hat, *_ = model(X)
                y_proba_list.append(y_proba)
                y_hat_list.append(y_hat)
            y_proba = torch.cat(y_proba_list, dim=0)
            y_hat = torch.cat(y_hat_list, dim=0)

            loss = loss_fn(y_proba, y_batch)
            loss.backward()

            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_accuracy += y_hat.eq(y_batch).detach().cpu().mean(dtype=float)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        print('Epoch: {:2d}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, train_loss, train_accuracy))
        return train_loss, train_accuracy


    def test():
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        test_loss = 0.
        test_accuracy = 0.
        y_list = []
        y_hat_list = []

        with torch.no_grad():
            cnt = np.zeros(4, dtype=int)
            for i, (X, y) in enumerate(test_loader):
                if cfg.use_cuda:
                    X, y = X.cuda(), y.cuda()

                y_proba, y_hat, *score = model(X)
                loss = loss_fn(y_proba, y)

                test_loss += loss.detach().cpu().item()
                test_accuracy += y_hat.eq(y).detach().cpu().mean(dtype=float)

                y = y.detach().cpu()[0]
                y_hat = y_hat.detach().cpu()[0]
                y_list.append(y)
                y_hat_list.append(y_hat)

                j = y
                k = cnt[j]
                if k < 10:
                    cnt[j] += 1
                    X = X.detach().cpu()[0]
                    A = score[0].detach().cpu()[0]
                    save_img(X, filename=f'img_{j}_{k}.csv', nrow=int(np.sqrt(cfg.dataset.bag_size)))
                    save_score(A, filename=f'score_{j}_{k}.csv', nrow=int(np.sqrt(cfg.dataset.bag_size)))
                    if cfg.model.name == 'additive':
                        P = score[1].detach().cpu()[0]
                        P = torch.transpose(P, 1, 0)
                        for l in range(cfg.model.num_classes):
                            save_score(P[l], filename=f'score_{j}_{k}_{l}.csv', nrow=int(np.sqrt(cfg.dataset.bag_size)))

        y = np.array(y_list)
        y_hat = np.array(y_hat_list)
        np.savetxt('y_true.csv', y, delimiter=',')
        np.savetxt('y_pred.csv', y_hat, delimiter=',')

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        print('Test Set, Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_accuracy))
        return test_loss, test_accuracy


    print('Start Training')
    train_loss_list = []
    train_accuracy_list = []
    for epoch in range(1, cfg.settings.epochs + 1):
        train_loss, train_accuracy = train(epoch)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
    train_loss = np.array(train_loss_list)
    train_accuracy = np.array(train_accuracy_list)
    np.savetxt('train_loss.csv', train_loss, delimiter=',')
    np.savetxt('train_accuracy.csv', train_accuracy, delimiter=',')
    torch.save(model.state_dict(), 'model_weights.pth')

    print('Start Testing')
    test()

    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()