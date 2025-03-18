import torch
import torch.nn as nn
import torchvision 
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define parameters
input_size = 784 #28x28 pixel
hidden_size = 100
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

#Importing MNIST Dataset 
train_data = datasets.MNIST(root='./Data', train = True, transform = transforms.ToTensor(), download= True)
test_data = datasets.MNIST(root='./Data', train = False, transform = transforms.ToTensor())

#Dataloader
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

examples = iter(train_loader)
samples, labels =next(examples)
print(samples.shape, labels.shape)

#ploting data
for i in range(6):
 plt.subplot(3,2, i+1)
 plt.imshow(samples[i][0], cmap='gray')
 plt.show()

# Now Classify the MNIST dataset
class NeuralNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
      super(NeuralNetwork, self).__init__()

# Creating Layers
      self.l1=nn.Linear(input_size, hidden_size)
      #Applying activation function
      self.relu = nn.ReLU()
      self.l2=nn.Linear(hidden_size, num_classes)

  def forward(self, x):
      out = self.l1(x)
      out = self.relu(out)
      out = self.l2(out)
      return out
  
model = NeuralNetwork(input_size, hidden_size, num_classes)
model = NeuralNetwork(input_size, hidden_size, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

#Traing Loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
   for i, (images, labels) in enumerate(train_loader):
      images = images.reshape(-1, 28*28).to(device)
      labels = labels.to(device)

# forward Pass
      outputs = model(images)
      loss = criterion(outputs, labels)


# Backward Pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i+1) % 100 == 0:
         print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}' )

# Testing 
with torch.no_grad():
   n_correct = 0
   n_samples = 0
   for images, labels in test_loader:
       images = images.reshape(-1, 28*28).to(device)
   labels = labels.to(device)

   #Prediction

   outputs = model(images)
   _, prediction = torch.max(outputs, 1)
   n_samples += labels.shape[0]
   n_correct += (prediction == labels).sum().item()

   Accuracy = 100.0*n_correct/n_samples

   print(f'Accuracy = {Accuracy}')


      
   



