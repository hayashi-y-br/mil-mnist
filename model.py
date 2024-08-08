import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, num_classes=4):
        super(Attention, self).__init__()
        self.num_classes = num_classes
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # Kx1
        A = torch.transpose(A, 1, 0)  # 1xK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # 1xM

        y_proba = self.classifier(Z)  #1xC
        y_hat = torch.argmax(y_proba, dim=1)

        return y_proba, y_hat, A


class Additive(nn.Module):
    def __init__(self, num_classes=4):
        super(Additive, self).__init__()
        self.num_classes = num_classes
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L, bias=False),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1, bias=False)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # Kx1
        A = F.softmax(A, dim=0)  # softmax over K

        Z = torch.mul(A, H)  # KxM

        P = self.classifier(Z)  # KxC
        P = P.unsqueeze(0)  # 1xKxC

        y_proba = torch.mean(P, dim=1)  # 1xC
        y_hat = torch.argmax(y_proba, dim=1)

        return y_proba, y_hat, torch.transpose(A, 1, 0), P


if __name__ == '__main__':
    X = torch.rand(16, 1, 28, 28)

    model = Attention()
    y_proba, y_hat, A = model(X)
    print(y_proba.shape, y_hat.shape, A.shape)

    model = Additive()
    y_proba, y_hat, A, P = model(X)
    print(y_proba.shape, y_hat.shape, A.shape, P.shape)