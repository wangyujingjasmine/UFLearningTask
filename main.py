import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import CIFAR10NET
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.onnx

writer = SummaryWriter('log')

best_acc = 0

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = CIFAR10NET()

if torch.cuda.is_available():
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

# dumy_input = Variable(torch.randn(128, 3, 32, 32)).cuda()
# torch.onnx.export(net, dumy_input, 'resnet.proto', verbose=True)
# writer.add_graph_onnx('resnet.proto')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# Training
def train(epoch):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        counter_num = batch_idx+len(trainloader)*epoch
        print(counter_num,
              batch_idx, len(trainloader),
              'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1),
               100.*correct/total, correct, total))

        writer.add_scalar('train/loss', train_loss / (batch_idx + 1), counter_num)
        writer.add_scalar('train/Acc', 1.*correct/total, counter_num)
        # for name, param in net.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), counter_num)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        counter_num = batch_idx+len(testloader)*epoch
        print(counter_num ,batch_idx, len(testloader), 'Text Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        writer.add_scalar('test/loss', test_loss / (batch_idx + 1), counter_num)
        writer.add_scalar('test/Acc', 1. * correct / total, counter_num)
if __name__ == '__main__':

    for epoch in range(0, 100):
        train(epoch)
        test(epoch)
    writer.close()
