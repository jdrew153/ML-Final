---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: venv
  language: python
  name: python3
---

```{code-cell} ipython3
## trying to rename the training image files to have unique numbering
import os

def count_files(directory):
    file_ct = 0
    for _, _, files in os.walk(directory):
        file_ct += len(files)
    return file_ct

training_img_dirs = ['broken_large', 'broken_small', 'contamination', 'good']

total_train_file_count = 0

for img_dir in training_img_dirs:
    full_path = os.path.join('bottle/train', img_dir)
    file_ct = count_files(full_path)
    print(f'{img_dir} has {file_ct} files')

    total_train_file_count += file_ct

print(f'Total training files: {total_train_file_count}')


for i in range(len(training_img_dirs)):
    img_dir = training_img_dirs[i]
    full_path = os.path.join('bottle/test', img_dir)
    for j, file in enumerate(os.listdir(full_path)):
        new_name = f'{img_dir}_{j}.jpg'
        os.rename(os.path.join(full_path, file), os.path.join(full_path, new_name))
        print(f'{file} -> {new_name}')

    
  
```

```{code-cell} ipython3
import os
import json

## Need to create the json file for the DataLoader

def create_json_labels(directory, output_file):
    ## remove an existing annotation file.
    if os.path.exists(output_file):
        os.remove(output_file)

    ## Created from the ground truth folder
    ## Key - filename (i.e img)
    label_img_mapping = {}
    dir = os.path.join('bottle', directory)
    for root, dirs, files in os.walk(dir):
        
        for file in files:
            if 'good' in file:
                label_img_mapping[file] = 'good'
            elif 'broken_large' in file:
                label_img_mapping[file] = 'broken_large'
            elif 'broken_small' in file:
                label_img_mapping[file] = 'broken_small'
            elif 'contamination' in file:
                label_img_mapping[file] = 'contamination'

    with open(output_file, 'w') as f:
        json.dump(label_img_mapping, f, indent=4)


## Create the annotations for the training dataset - broken_large, broken_small, contamination, good
create_json_labels('train', 'train_annotations.json')            

## Create the annotations for the test dataset - good
create_json_labels('test', 'test_annotations.json')
```

```{code-cell} ipython3
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2


class CustomBottleDataset(Dataset):
    def __init__(self, data_directory, labels_path, transform = None) -> None:

        with open(labels_path, 'r') as f:
            self.image_labels = json.load(f)

        self.data_dir = data_directory
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_labels)
    
    def __getitem__(self, index):
        filename = list(self.image_labels.keys())[index]
        label = self.image_labels[filename]

        img_path = os.path.join(self.data_dir, filename)
        img = read_image(img_path) / 255.0

        if self.transform:
            img = self.transform(img)
        
        label_mapping = {'good' : 0, 'broken_small' : 1, 'broken_large' : 2, 'contamination' : 3}
        label = label_mapping[label]

        return img, label

T = v2.Compose([
      v2.Resize(size=(224,224), antialias=True),
      v2.RandomRotation(degrees=45),
      v2.RandomHorizontalFlip(),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomBottleDataset(data_directory='bottle/train', labels_path='train_annotations.json', transform=T)
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
```

```{code-cell} ipython3
import torchvision

model = torchvision.models.resnet18(pretrained=True)

print(list(model.__dict__["_modules"].keys()))
```

```{code-cell} ipython3
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

class DefectIdentificationNetwork(nn.Module):
    def __init__(self):
        super(DefectIdentificationNetwork, self).__init__()
        ref_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = ref_model.conv1
        self.bn1 = ref_model.bn1
        self.relu = ref_model.relu
        self.maxpool = ref_model.maxpool
        self.layer1 = ref_model.layer1
        self.layer2 = ref_model.layer2
        self.layer3 = ref_model.layer3 
        self.layer4 = ref_model.layer4
        self.avgpool = ref_model.avgpool
        

        ## custom fc for # of classes i want to classify -> currently at 512 (classifying 4 attribute)
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

```{code-cell} ipython3
import pandas as pd

model = DefectIdentificationNetwork()


if os.path.exists('models/base-model.pth'):
    print(f'Using saved model.')
    model.load_state_dict(torch.load('models/base-model.pth'))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Type of device used for training: {device}')
model = model.to(device)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 10
model.train() # Put network into training mode, where neural network weights can change

loss_arr = []

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item()}')
        
        torch.save(model.state_dict(), 'models/base-model.pth')
        


    avg_loss = total_loss / len(train_dataloader)
    loss_arr.append(avg_loss)
    print(f'Average Loss after epoch {epoch + 1}: {avg_loss}')

loss_tracker_base_path = "reports/loss_tracker/loss_per_epoch_batch_base.csv"

base_loss_tracker = None

if os.path.exists(loss_tracker_base_path):
    base_loss_tracker = pd.read_csv(loss_tracker_base_path)
else:
    base_loss_tracker = pd.DataFrame({"Loss" : []})

## loss tracker example format - "Loss" : [0.5, 0.3, 0.2]
average_loss = sum(loss_arr) / num_epochs
base_loss_tracker = pd.concat([base_loss_tracker, pd.DataFrame({"Loss": [average_loss]})], ignore_index=True)

base_loss_tracker.to_csv(loss_tracker_base_path, index=False)

plt.plot(loss_arr)
plt.title('Loss Over Time - Base Images')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Base-loss.png')
```

```{code-cell} ipython3
from sklearn.metrics import accuracy_score
import cv2 
import pandas as pd

T = v2.Compose([
      v2.Resize(size=(224,224), antialias=True),
      v2.RandomRotation(degrees=15),
      v2.RandomHorizontalFlip(p=0.9),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = CustomBottleDataset(data_directory='bottle/test', labels_path='test_annotations.json', transform=T)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


predictions = []
groundtruth = []

model = DefectIdentificationNetwork()

if not os.path.exists('models/base-model.pth'):
    print(f'Execute the previous cell to train the base model.')

model.load_state_dict(torch.load('models/base-model.pth', map_location=device))
model.to(device)

model.eval()  # Ensure the model is in evaluation mode


# Reverse the label mapping for easier interpretation
label_mapping = {'good': 0, 'broken-small': 1, 'broken-large': 2, 'contamination': 3}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

filenames = list(test_dataset.image_labels.keys())

global_idx = 0

conf_matrix_mapping = {
    "Image" : [],
    "Actual" : [],
    "Predicted" : [],
    "Result" : []
}

for batch_idx, data in enumerate(test_loader):
    images, labels = data
    if torch.mps.is_available():
        images = images.to('mps')
        labels = labels.to('mps')

    # Perform forward pass without disabling gradients
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    predictions.extend(predicted.cpu().numpy())
    groundtruth.extend(labels.cpu().numpy())

    for i in range(len(images)):
        image = images[i]
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        # Write classification results to the file
        with open("classification_results.txt", "a") as log_file:
            log_file.write(
                f"Image: {filenames[global_idx]}, True: {reverse_label_mapping[true_label]}, Pred: {reverse_label_mapping[pred_label]}\n"
            )

        # Reverse normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        image = image * std[:, None, None] + mean[:, None, None]

        # Plot if prediction is incorrect
        if pred_label != true_label:
            plt.imshow(image.cpu().permute(1, 2, 0).clip(0, 1).numpy())
            plt.title(f"Filename: {filenames[global_idx]} True: {reverse_label_mapping[true_label]}, Pred: {reverse_label_mapping[pred_label]}")
            plt.axis('off')
            plt.show()
        
        conf_matrix_mapping['Image'].append(filenames[global_idx])
        conf_matrix_mapping['Actual'].append(reverse_label_mapping[true_label])
        conf_matrix_mapping['Predicted'].append(reverse_label_mapping[pred_label])
        conf_matrix_mapping['Result'].append(pred_label == true_label)

        global_idx += 1  # Increment global index

# Calculate accuracy
accuracy = accuracy_score(groundtruth, predictions)
print(f'Accuracy: {accuracy * 100}%')

conf_df = pd.DataFrame(conf_matrix_mapping)

report_path = "reports/base_conf_matrix.csv"

conf_df.to_csv(report_path)

# Count the number of False results
num_false = (conf_df['Result'] == False).sum()

# Total number of rows
total = len(conf_df)

# Calculate misclassification rate
misclassification_rate = num_false / total

print("Number of false classifications:", num_false)
print("Misclassification rate: %", (misclassification_rate * 100))
```

```{code-cell} ipython3
## Sharpening images 

import cv2


## sharpening all training images and moving them to sharpened directory

def sharpen_images(base_dir, new_dir):
    # Ensure the output directory exists
    os.makedirs(new_dir, exist_ok=True)

    # Sobel kernel will be used for edge detection
    ct = 0  # Counter for processed images

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            base_image_path = os.path.join(root, file)

            if os.path.exists(base_image_path):
                # Load the original color image
                base_img = cv2.imread(base_image_path)
                if base_img is None:
                    print(f"Could not read the image at path: {base_image_path}")
                    continue

                # Convert the color image to grayscale for edge detection
                base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

                # Apply Sobel filtering to detect edges
                sobel_x = cv2.Sobel(base_img_gray, cv2.CV_64F, 1, 0, ksize=5)
                sobel_y = cv2.Sobel(base_img_gray, cv2.CV_64F, 0, 1, ksize=5)
                edges = cv2.magnitude(sobel_x, sobel_y)

                # Normalize edges to the range [0, 255] and convert to uint8
                edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                # Convert edges to 3-channel format for overlay
                edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                # Overlay the edges onto the original color image
                sharpened_color_img = cv2.addWeighted(base_img, 1, edges_3channel, 1.0, 0)

                # Save the sharpened image
                sharpened_img_path = os.path.join(new_dir, file)
                cv2.imwrite(sharpened_img_path, sharpened_color_img)
                print(f"Saved sharpened image to: {sharpened_img_path}")
                ct += 1
            else:
                print(f"Base image path does not exist: {base_image_path}")

    print(f"Sharpened {ct} images")


# Directories for training and testing data
base_training_directory = 'bottle/train/'
sharpened_training_directory = 'bottle/sharpened/train'

if not os.path.exists(sharpened_training_directory):
    sharpen_images(base_training_directory, sharpened_training_directory)
else:
    print('Skipping creating training images.')

base_test_directory = 'bottle/test/'
sharpened_test_directory = 'bottle/sharpened/test'

if not os.path.exists(sharpened_test_directory):
    sharpen_images(base_test_directory, sharpened_test_directory)
else:
    print('Skipping creating testing images.')


## create the annotation files for the sharpened images.
create_json_labels('sharpened/train', 'bottle/sharpened/train_annotations.json')            

## Create the annotations for the test dataset - good
create_json_labels('sharpened/test', 'bottle/sharpened/test_annotations.json')
```

```{code-cell} ipython3
## train the model on the sharpened images

T = v2.Compose([
      v2.Resize(size=(224,224), antialias=True),
      v2.RandomRotation(degrees=45),
      v2.RandomHorizontalFlip(),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sharpened_train_dataset = CustomBottleDataset(data_directory='bottle/sharpened/train', labels_path='bottle/sharpened/train_annotations.json', transform=T)
sharpened_train_dataloader = DataLoader(sharpened_train_dataset, batch_size=5, shuffle=True)


model = DefectIdentificationNetwork()

sharpened_model_path = 'models/sharpened-model.pth'

if os.path.exists(sharpened_model_path):
    print(f'Using saved model.')
    model.load_state_dict(torch.load(sharpened_model_path))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Type of device used for training: {device}')
model = model.to(device)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 10
model.train() # Put network into training mode, where neural network weights can change

loss_arr = []

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch_idx, (images, labels) in enumerate(sharpened_train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], Loss: {loss.item()}')
        
        torch.save(model.state_dict(), sharpened_model_path)


    avg_loss = total_loss / len(train_dataloader)
    loss_arr.append(avg_loss)
    print(f'Average Loss after epoch {epoch + 1}: {avg_loss}')

loss_tracker_base_path = "reports/loss_tracker/loss_per_epoch_batch_sharpened.csv"

sharpened_loss_tracker = None

if os.path.exists(loss_tracker_base_path):
    sharpened_loss_tracker = pd.read_csv(loss_tracker_base_path)
else:
    sharpened_loss_tracker = pd.DataFrame({"Loss" : []})

## loss tracker example format - "Loss" : [0.5, 0.3, 0.2]
average_loss = sum(loss_arr) / num_epochs
sharpened_loss_tracker = pd.concat([sharpened_loss_tracker, pd.DataFrame({"Loss": [average_loss]})], ignore_index=True)

sharpened_loss_tracker.to_csv(loss_tracker_base_path, index=False)

plt.plot(loss_arr)
plt.title('Loss Over Time - Sharpened Images')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Sharpened-loss.png')
```

```{code-cell} ipython3
from sklearn.metrics import accuracy_score
import cv2 

T = v2.Compose([
      v2.Resize(size=(224,224), antialias=True),
      v2.RandomRotation(degrees=15),
      v2.RandomHorizontalFlip(p=0.9),
      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

sharpened_test_dataset = CustomBottleDataset(data_directory='bottle/sharpened/test', labels_path='bottle/sharpened/test_annotations.json', transform=T)
sharpened_test_loader = DataLoader(sharpened_test_dataset, batch_size=5, shuffle=False)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import pandas as pd


predictions = []
groundtruth = []

model = DefectIdentificationNetwork()

model.load_state_dict(torch.load(sharpened_model_path, map_location=device))
model.to(device)

model.eval()  # Ensure the model is in evaluation mode


# Reverse the label mapping for easier interpretation
label_mapping = {'good': 0, 'broken-small': 1, 'broken-large': 2, 'contamination': 3}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

filenames = list(sharpened_test_dataset.image_labels.keys())

global_idx = 0

conf_matrix_mapping = {
    "Image" : [],
    "Actual" : [],
    "Predicted" : [],
    "Result" : []
}

for batch_idx, data in enumerate(sharpened_test_loader):
    images, labels = data
    if torch.mps.is_available():
        images = images.to('mps')
        labels = labels.to('mps')

    # Perform forward pass without disabling gradients
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    predictions.extend(predicted.cpu().numpy())
    groundtruth.extend(labels.cpu().numpy())

    for i in range(len(images)):
        image = images[i]
        true_label = labels[i].item()
        pred_label = predicted[i].item()

        # Write classification results to the file
        with open("classification_results.txt", "a") as log_file:
            log_file.write(
                f"Image: {filenames[global_idx]}, True: {reverse_label_mapping[true_label]}, Pred: {reverse_label_mapping[pred_label]}\n"
            )

        # Reverse normalization for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        image = image * std[:, None, None] + mean[:, None, None]

        # Plot if prediction is incorrect
        if pred_label != true_label:

            plt.imshow(image.cpu().permute(1, 2, 0).clip(0, 1).numpy())
            plt.title(f"Filename: {filenames[global_idx]} True: {reverse_label_mapping[true_label]}, Pred: {reverse_label_mapping[pred_label]}")
            plt.axis('off')
            plt.show()
       
        conf_matrix_mapping['Image'].append(filenames[global_idx])
        conf_matrix_mapping['Actual'].append(reverse_label_mapping[true_label])
        conf_matrix_mapping['Predicted'].append(reverse_label_mapping[pred_label])
        conf_matrix_mapping['Result'].append(pred_label == true_label)

        global_idx += 1  # Increment global index

# Calculate accuracy
accuracy = accuracy_score(groundtruth, predictions)
print(f'Accuracy: {accuracy * 100}%')

conf_df = pd.DataFrame(conf_matrix_mapping)

report_path = "reports/sharpened_conf_matrix.csv"

conf_df.to_csv(report_path)

# Count the number of False results
num_false = (conf_df['Result'] == False).sum()

# Total number of rows
total = len(conf_df)

# Calculate misclassification rate
misclassification_rate = num_false / total

print("Number of false classifications:", num_false)
print("Misclassification rate: %", (misclassification_rate * 100))
```
