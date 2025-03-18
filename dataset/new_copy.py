import torch
import torchvision.transforms as transforms
from PIL import Image
from new import NeuralNetwork

# Load the trained model
model = NeuralNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))  # Load the trained model's state dict
model.eval()  # Set the model to evaluation mode

# Prepare the image you want to test
image_path = 'test_image.png'
image = Image.open(image_path)

# Preprocess the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize as per MNIST dataset mean and standard deviation
])
input_image = transform(image)
input_image = input_image.unsqueeze(0)  # Add batch dimension

# Pass the preprocessed image through the model
with torch.no_grad():
    output = model(input_image)

# Interpret the model's output
_, predicted_class = torch.max(output, 1)

print(f'Predicted class: {predicted_class.item()}')
