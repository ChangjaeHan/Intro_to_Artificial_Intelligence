import operator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.
# Author: Changjae Han


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT:
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set=datasets.FashionMNIST('./data', train=True,
            download=True,transform=custom_transform)

    test_set=datasets.FashionMNIST('./data', train=False,
            transform=custom_transform)

    if training == True :
        loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size = 64)

    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT:
        None

    RETURNS:
        An untrained neural network model
    """
   
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10)
        )

    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """
    
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    #iterate epoch
    for epoch in range(T):
        total = 0
        correct = 0
        running_loss = 0.0
        for data in train_loader:
            images, labels = data
            opt.zero_grad()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            #calculate loss*batch
            running_loss += loss.item()*64
            
        running_loss = running_loss/total
        print("Train Epoch:", epoch, "  Accuracy: {}/{}" .format(correct,total),"({:.2f}%)".format(100*correct/total), "Loss: {:.3f}".format(running_loss))


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT:
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    #similar approach to iterate but no gradient
    with torch.no_grad():
        for data, labels in test_loader:
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()*64

        running_loss = running_loss/total

    if(show_loss == False):
        print("Accuracy: {:.2f}%".format(100*correct/total))
    else:
        print("Average loss: {:.4f}".format(running_loss))
        print("Accuracy: {:.2f}%".format(100*correct/total))
 

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    prob = F.softmax(model(test_images), dim=None)
    classN = prob[index].tolist()

    #dictionary to tag key-value to return
    dic = {}
    dic[classN[0]] = "T-shirt/top"
    dic[classN[1]] = "Trouser"
    dic[classN[2]] = "Pullover"
    dic[classN[3]] = "Dress"
    dic[classN[4]] = "Coat"
    dic[classN[5]] = "Sandal"
    dic[classN[6]] = "Shirt"
    dic[classN[7]] = "Sneaker"
    dic[classN[8]] = "Bag"
    dic[classN[9]] = "Ankle Boot"

    sortD = sorted(dic.items(),key=operator.itemgetter(0), reverse=True)

    print(sortD[0][1],": {:.2f}%".format(sortD[0][0]*100))
    print(sortD[1][1],": {:.2f}%".format(sortD[1][0]*100))
    print(sortD[2][1],": {:.2f}%".format(sortD[2][0]*100))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions.
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    print(test_loader.dataset)
    model = build_model()
    print(model)
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    pred_set, _ = next(iter(test_loader))
    predict_label(model, pred_set, 1)
