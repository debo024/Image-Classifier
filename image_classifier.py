import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

transform= transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,),(0.5,)),
                              ])
trainset= datasets.MNIST('~/.pytorch/MNIST_data/', download= True, train= True, transform=transform)
trainloader= torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


model = nn.Sequential(nn.Linear(784,128),
                     nn.ReLU(),
                     nn.Linear(128,64),
                     nn.ReLU(),
                     nn.Linear(64,10),
                     nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps= model(images)
loss= criterion(logps,labels)

print(loss)


# print('Before \n', model[0].weight.grad)
# loss.backward()
# print('After \n', model[0].weight.grad)


from torch import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)


# print('Initial weights - ', model[0].weight)

# images, labels = next(iter(trainloader))
# images.resize_(64,784)

# optimizer.zero_grad()

# output = model.forward(images)
# loss = criterion(output, labels)
# loss.backward()
# print('Gradient - ', model[0].weight.grad)



model = nn.Sequential(nn.Linear(784,128),
                     nn.ReLU(),
                     nn.Linear(128,64),
                     nn.ReLU(),
                     nn.Linear(64,10),
                     nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
            



%matplotlib inline
from urllib.request import urlretrieve
urlretrieve('https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/3bd7dea850e936d8cb44adda8200e4e2b5d627e3/intro-to-pytorch/helper.py', 'helpers.py')
import helpers

images, labels = next(iter(trainloader))
print(helper)

img = images[0].view(1, 784)
with torch.no_grad():
    logits = model.forward(img)
    
    
ps = F.softmax(logits, dim=1)
helpers.view_classify(img.view(1,28,28), ps)