import numpy as np
import torch
from torch._C import LongStorageBase
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

training_set = datasets.MNIST(r'./training_set', download=True, train=True, transform=transform)
validation_set = datasets.MNIST(r'./validation_set', download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]), 
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)

# print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)

loss.backward()
optimizer = optim.SGD(model.parameters(), lr = 0.003, momentum=0.9)
time0 = time()
epochs = 15

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    else:
        print("epoch: {} \nloss: {}".format(e, running_loss/len(trainloader)))

print("time (min) taken: {}".format((time() - time0)/60))

correct_predictions = 0
all_predictions = 0

for images,labels in validloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad:
            logps = model(img)
        ps = torch.exp(logps)
        probability = list(ps.numpy()[0])
        prediction_label = probability.index(max(probability))
        actual_label = labels.numpy()[i]
        if prediction_label==actual_label:
            correct_predictions += 1
        all_predictions += 1
 
print("images tested: {}".format(all_predictions))
print("accuracy: {}%".format((correct_predictions/all_predictions)*100))

torch.save(model, r'./mnist_model.pt')