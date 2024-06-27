import torch
from torch import nn


class RouterNN(nn.Module):

    def __init__(self, in_channels, n_experts, conv_size=128):
        super(RouterNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=conv_size,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm1d(conv_size)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv1d(in_channels=conv_size,
                               out_channels=conv_size,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(conv_size)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv1d(in_channels=conv_size,
                               out_channels=n_experts,
                               kernel_size=3,
                               padding=1)
        self.bn3 = nn.BatchNorm1d(n_experts)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)
        # import pdb
        # pdb.set_trace()
        # x = x.max(dim=2)[0]
        x = x.sum(dim=2)
        x = self.softmax(x)
        # print(x)
        return x


class TrainRouterNN(nn.Module):

    def __init__(self, configs):
        super(TrainRouterNN, self).__init__()
        in_channels = configs['dataset_conf']['fbank_conf']['num_mel_bins']
        n_experts = configs['model_conf']['n_experts']

        self.router = RouterNN(in_channels, n_experts, conv_size=128)
        self.loss_computation = nn.CrossEntropyLoss()

    def forward(self, batch, device):
        # import pdb
        # pdb.set_trace()
        speech = batch['feats'].to(device)
        target = batch['target'].squeeze(1).to(device)
        out_router = self.router(speech)

        # try:
        loss = self.loss_computation(out_router, target)
        # except:
        #     # import pdb
        #     pdb.set_trace()
        #     print(out_router.size(), target.size())
        #     raise
        th_accuracy = sum(
            torch.argmax(out_router, dim=1) == target) / len(target)
        with open('save_vals.txt', 'a') as f:
            [
                f.write(
                    str(out_router[i].detach().cpu().numpy()) + '|' +
                    str(target[i].detach().cpu().numpy()) + '\n')
                for i in range(len(target))
            ]

        return {
            "loss": loss,
            "th_accuracy": th_accuracy,
        }

    def decode(self, batch, device):
        speech = batch['feats'].to(device)
        out_router = self.router(speech)
        return torch.argmax(out_router).item()
