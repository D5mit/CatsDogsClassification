import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'ImagesCatDog'


class myNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def validation(model, testloader, criterion, inputsize):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        images = images.resize_(images.size()[0], inputsize)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy


def modelTrain(model, trainloader, testloader, criterion, optimizer, inputsize, epochs=5, print_every=40):

    print('Start Training....')

    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1

            # Flatten images into a input size long vector
            images.resize_(images.size()[0], inputsize)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion, inputsize)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = myNetwork(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model

# Main Program
# Define transforms for the training data and testing data
print('Cat and Dog Image training using PyTorch')
print('----------------------------------------')

train_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
print('Training data directory:' + data_dir )
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

images, labels = next(iter(trainloader))
print('TrainLoaded Images input size:' + str(images.shape))
print('TrainLoaded Labels input size:' + str(images.shape))

# Create the network, define the criterion and optimizer
inputsize = 1024
outputsize = 2
imodel = myNetwork(inputsize, outputsize, [512, 256, 128])
criterion = nn.NLLLoss()
optimizer = optim.Adam(imodel.parameters(), lr=0.001)

print('Training Model:')
print(imodel)

# train the model
modelTrain(imodel, trainloader, testloader, criterion, optimizer, inputsize, epochs=1)

print('')
print('Model after training:')
print("Our model: \n\n", imodel, '\n')
print("The state dict keys: \n\n", imodel.state_dict().keys())


# save and reload model
# Todo must still test
checkpoint = {'input_size': 2014,
              'output_size': 2,
              'hidden_layers': [each.out_features for each in imodel.hidden_layers],
              'state_dict': imodel.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

model = load_checkpoint('checkpoint.pth')
print(model)


# test and use model

# predict
images, labels = next(iter(testloader))

# 0
plt.imshow(images[0].numpy().squeeze(), cmap='Greys_r');
plt.show()

img = images[0].view(1, inputsize)

# Turn off gradients to speed up this part
with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)

# print prediction
print(np.around(ps, decimals=3))

prednr = np.argmax(ps)
print(prednr)
print(labels[0])