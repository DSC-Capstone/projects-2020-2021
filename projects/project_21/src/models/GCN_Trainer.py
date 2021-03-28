import torch.optim as optim
import time
from src.models.GCN import *

class GCN_Trainer(object):
    def __init__(self, features, adj, labels):
        #do something

        #this is used to help the test set training
        shape_idx = 1
        if len(features.shape) == 1:
            shape_idx = 0


        self.model = GCN(nfeat=features.shape[shape_idx],
                        nhid=16,
                        nclass=len(labels.unique()) + 1,
                        dropout=0.5)
        self.optimizer = optim.Adam(self.model.parameters(),
                                   lr=.01, weight_decay=5e-4)
        self.features = features
        self.adj = adj
        self.labels = labels


    def train(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)

        print(output.shape, self.labels.shape)

        loss = nn.CrossEntropyLoss()
        loss_train = loss(output, self.labels.type(torch.LongTensor))
        acc_train = self.accuracy(output, self.labels)
        loss_train.backward()
        self.optimizer.step()


        loss_val = loss(output, self.labels.type(torch.LongTensor))
        acc_val = self.accuracy(output, self.labels)

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test(self):
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss = nn.CrossEntropyLoss()
        loss_test = loss(output, self.labels.type(torch.LongTensor))
        acc_test = self.accuracy(output, self.labels)

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def complete_train(self):
        t_total = time.time()
        for epoch in range(100):
            self.train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        self.test()

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)
