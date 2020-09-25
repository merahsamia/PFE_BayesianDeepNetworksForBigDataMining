#Importing necessary Libraries
import torch
import torchvision
import torchvision.transforms as transforms  
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys

def config(batch_size): 
  
  configuration=[]
  #convert the dataset 
  train_transform = transforms.Compose(
   
      [transforms.ToTensor(),#transform the dataset into PyTorch Tensor
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#normalize pixel values

  #transformation on the test set
  test_transform = transforms.Compose(
      [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])




  #downloading the train_valid  sets
  train_valid = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)


  #splitting train_valid to trainset and validset
  trainset, validset = torch.utils.data.random_split(train_valid, [45000, 5000])

  #loading the train set
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
  #loading the validation set
  validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

  #downloading the test set
  testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=test_transform)

  #loading the test set
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


  configuration.extend([trainset,validset,testset,trainloader,validloader,testloader])
  
  return configuration

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def classes_freq():
  
  labels = torch.tensor([instance[1] for instance in trainset])
  class_freq = labels.bincount()

  indexes = np.arange(len(classes))
  width = 0.6

  plt.bar(indexes, class_freq.numpy(), align='edge', width=width , color=['crimson','darkorchid'])
  plt.xticks(indexes + width * 0.5, classes)
  plt.title('CIFAR10 classes frequency')
  plt.show()

def show_random_iamges(dataiter):
  
  images, labels = dataiter.next()
  print(images.shape)
  # show images
  imshow(torchvision.utils.make_grid(images[:4]))
  # print labels
  print('      '.join('%5s' % classes[labels[j]] for j in range(4)))

def plotting (num_epochs,loss_acc1,loss_acc2,Acc):
    x= np.linspace(1, num_epochs , num_epochs).astype(int)#defining the x-axis  
    fig, ax1 = plt.subplots() # Create a figure and only one subplot using ax1
    

    
    ax1.set_xlabel('epochs')  
    ax1.plot(x, loss_acc1, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # we already handled the x-label with ax1
    ax2.plot(x, loss_acc2, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
   

    # Add title
    if Acc == True:
      plt.title('Model Loss ',pad=30)
      ax1.set_ylabel('Train Loss', color='red')
      ax2.set_ylabel('Valid Loss', color='blue')
    else:
      plt.title('Model Accuracy',pad=30)
      ax1.set_ylabel(' Train Accuracy', color='red')
      ax2.set_ylabel(' Valid Accuracy', color='blue')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(192, 192, 5, padding=2)
        self.fc1 = nn.Linear(192 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.3)
        
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # recommended to add the relu
        x = self.dropout(x)  
        x = self.pool(F.relu(self.conv2(x)))  # recommended to add the relu
        x = self.dropout(x)
        x = x.view(-1, 192 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(self.dropout(x)) # no activation function needed for the last layer 
        return x

def train(num_epochs,criterion,optimizer,model,loss_acc,n_total_steps): 
    loss_vals=  []
    acc_vals = []
    loss_valid =[]
    acc_valid = []

    for epoch in range(num_epochs): #loop over the dataset multiple times

        n_correct = 0 #initialize number of correct predictions
        total = 0 
        acc= 0  #initialize accuracy of each epoch
        somme=0 #initialize somme of losses of each epoch
        epoch_loss= []
              
        for i, (images, labels) in enumerate(trainloader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model.train()(images)
            loss = criterion(outputs, labels)
           
            # Backward and optimize           
            optimizer.zero_grad() # zero the parameter gradients
            loss.backward()
            epoch_loss.append(loss.item())#add the loss to epoch_loss list
            optimizer.step()
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        
            # print statistics
            if (i+1) % 2000 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
  

        somme=(sum(epoch_loss))/len(epoch_loss)       
        loss_vals.append(somme) #add the training loss to loss_vals
        print(f'Training loss = {somme:.3f} ')    
        acc = 100 * n_correct / total
        acc_vals.append(acc) #add the training Accuracy to acc_vals
        print(f'Training Accuracy = {acc:.3f} %')

        n_correct = 0
        total = 0 
        acc= 0  
        somme=0 
        epoch_loss= []
        with torch.no_grad():
            for images, labels in validloader:
              images = images.to(device)
              labels = labels.to(device)
              #testing the model 
              outputs = model.eval()(images)
              loss = criterion(outputs, labels)
              epoch_loss.append(loss.item())
              # max returns (value ,index)
              _, predicted = torch.max(outputs, 1)
              total += labels.size(0)
              n_correct += (predicted == labels).sum().item()

            
        somme=(sum(epoch_loss))/len(epoch_loss)       
        loss_valid.append(somme) 
        print(f'Validation loss = {somme:.3f}')    
        acc = 100 * n_correct / total
        acc_valid.append(acc) 
        print(f'Validation Accuracy = {acc:.3f} %')
        
        
    #SAVE               
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)
    
    loss_acc.extend([loss_vals,acc_vals,loss_valid,acc_valid])

    return loss_acc

def test (batch_size,model):
           
  n_correct = 0
  n_samples = 0
  n_class_correct = [0 for i in range(10)]
  n_class_samples = [0 for i in range(10)]
    
  for images, labels in testloader:
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():#without calculating gradients
      #testing the model 
      outputs = model.eval()(images)

    # max returns (value ,index)
    _, predicted = torch.max(outputs, 1)
    n_samples += labels.size(0)
    n_correct += (predicted == labels).sum().item()
            
                
                
    for i in range(batch_size):
      label = labels[i]
      pred = predicted[i]
            
      if (label == pred):
        n_class_correct[label] += 1
      n_class_samples[label] += 1
        
        
        
  acc = 100.0 * n_correct / n_samples
  print(f'the score test is : {acc} %')
    
  #Accuracy for each class 
  classes_acc = []
  for i in range(10):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #print(f'Accuracy of {classes[i]}: {acc} %')
    classes_acc.append(acc)
            
  return classes_acc

def plot_uncertainty(proportions):

  colors = ['b','mediumspringgreen','r','lightsalmon','orange','steelblue','deepskyblue','lightslategray','teal','chocolate']
  
  plt.pie(proportions, colors=colors,
        startangle=20, shadow=True,
        radius=1.2, autopct='%1.1f%%')


  plt.title('uncertainty of different classes ',pad=30)
  
  plt.legend(labels=classes,loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
  plt.show()

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def plot_classes_acc(accuracy):
  y_pos = np.arange(len(accuracy))

  # Create horizontal bars
  plt.barh(classes, accuracy,color=['b','mediumspringgreen','r','lightsalmon','orange','steelblue','deepskyblue','lightslategray','teal','chocolate'])
 
  # Create names on the y-axis
  plt.yticks(y_pos, classes)

  #set title
  plt.title('Classes Accuracy using MC Dropout',pad=30)
 
  # Show graphic
  plt.show()

def mcdropout_test(batch_size,n_classes,model,T):

    #set non-dropout layers to eval mode
    model.eval()

    #set dropout layers to train mode
    enable_dropout(model)

    acc = 0
    correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    softmax = nn.Softmax(dim=1)
    classes_mean = []
       
    for images,labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        classes_mean_batch = []
            
        with torch.no_grad():
          output_list = []
          
          #getting outputs for T forward passes
          for i in range(T):
            output = model(images)
            output = softmax(output)
            output_list.append(torch.unsqueeze(output, 0))
            
        
        concat_output = torch.cat(output_list,0)
        #print(concat_output)
  	
	      # getting mean of each class per batch across multiple MCD forward passes
        for i in range (n_classes):
          mean = torch.mean(concat_output[:, : , i])
          classes_mean_batch.append(mean)
        
	      # getting mean of each class for the testloader  
        classes_mean.append(torch.stack(classes_mean_batch))
        #print(classes_mean)
 
        #calculating mean for prediction 
        output_mean = concat_output.mean(0)      
          
        _, predicted  = torch.max(output_mean, 1) # get the index of the max log-probability
        correct += (predicted == labels).sum().item() #sum up correct predictions
        n_samples += labels.size(0)

        for i in range(batch_size):
            label = labels[i]
            predi =  predicted[i]
            
            if (label == predi):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        
        

    total_mean = []
    #print(len(classes_mean))

    concat_classes_mean = torch.stack(classes_mean)

    for i in range (n_classes):
      concat_classes = concat_classes_mean[: , i]
      total_mean.append(concat_classes)

    total_mean = torch.stack(total_mean)
    
    total_mean = np.asarray(total_mean.cpu())
    

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = (- np.sum(total_mean*np.log(total_mean + epsilon), axis=-1)).tolist()
    
    
    #calculate accuracy of the test
    acc = 100.0 * correct / n_samples
    print(f'the mcdropout score test is : {acc} %')

    #Accuracy for each class 
    acc_classes = []
    acc = 0
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        acc_classes .append(acc)

    plot_classes_acc(acc_classes)
    plot_uncertainty(entropy)

def main():
  global device
  global classes
  global testloader
  global trainset 
  global validset 
  global testset 
  global trainloader 
  global validloader 
  global testloader 

  # Device configuration
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Hyper-parameters
  num_epochs = 30
  batch_size = 4
  learning_rate = 0.001

  

  #Dataset configuration
  print("\n   ~~~~~~~~~ ****** ~~~~~~~~~ preparing Data ~~~~~~~~~ ****** ~~~~~~~~~") 
  print("\n")
  trainset,validset,testset,trainloader,validloader,testloader= config(batch_size)

  
  classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  print("printing sizes of train/validation/test sets")
  print('train set size: {}'.format(len(trainset)))
  print('validation set size: {}'.format(len(validset)))
  print('test set size: {}'.format(len(testset)))
  

  #initialising variables
  dataiter = iter(trainloader)
  n_total_steps = len(dataiter)#45/4 : batch_size= 11250
  loss_acc = []
  class_acc_test = []
  class_acc_mcdo = []


  #Data visualisation

  print("\n visualising classes frequency in CIFAR10 dataset")
  classes_freq()


  #print("\n please enter an image index between 0 and 44 999")
  #a = int(input())
     
     
  #img,label = trainset[a]
  #imshow(img)
  #print("image of  *****",classes [label],"*****")

  print("\n press 'y' if you want to see random training images ")
  b = input()
  if b == 'y':
    show_random_iamges(dataiter)

  
  # prepare the model
  model = Net().to(device)

  #Defining the Loss Function and Optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  
  print("\n _________________________________________________________________________________")
  print("\n     ~~~~~~~~~ ****** ~~~~~~~~~ start training ~~~~~~~~~ ****** ~~~~~~~~~")
  print("\n")
  loss_acc.extend(train(num_epochs,criterion,optimizer,model,loss_acc,n_total_steps))
  print('Finished Training')
  print("\n ~~~~~~******~~~~~~ plotting graphs of loss and accuracy ~~~~~~******~~~~~~")
  
  plotting(num_epochs,loss_acc[0],loss_acc[2],True)
  plotting(num_epochs,loss_acc[1],loss_acc[3],False)  
  print("\n _________________________________________________________________________________")
  
   
  print("\n     ~~~~~~~~~ ****** ~~~~~~~~~ start testing with MCDO ~~~~~~~~~ ****** ~~~~~~~~~")
  T = 2
  mcdropout_test(batch_size,10,model,T)
  print("finished testing")
  print("_________________________________________________________________________________")

  


