import torch
import torchvision.transforms as transforms
from PIL import Image
# pip install efficientnet_pytorch

# Assuming 'mymodel_checkpoint.pth' contains your pre-trained model
mymodel = torch.load('D:\ML_projects\DR web\mymodel_checkpoint.pth')
mymodel.eval()

# Define transformations for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to match model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Load the image
image_path = 'D:\ML_projects\DR web\DR_severe.png'  # Change this to your image path
image = Image.open(image_path)

# Preprocess the image
input_image = transform(image).unsqueeze(0)  # Add batch dimension

# Pass the preprocessed image through the model
with torch.no_grad():
    output = mymodel(input_image)

# Interpret the output to get predictions
predicted_class = torch.argmax(output, dim=1).item()

# Print the predicted class
print("Predicted Class:", predicted_class)
