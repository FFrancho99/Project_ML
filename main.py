import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from Model import Generator
from Model import Discriminator



# Parameters
batch_size = 8
num_epochs = 2
device = 'cuda:0'
print(torch.cuda.is_available())
print(torch.__version__)
device = 'cpu'
num_classes = 10
noise_var = 0.5

# Load dataset
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5), (0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
valset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)

print(trainset)

# Create dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize to show images correctly
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Print some samples of dataset as a sanity check

# Get some random training images
dataiter = iter(trainloader)
example_images, example_labels = next(dataiter)

print(example_images.shape)

# Show images
imshow(torchvision.utils.make_grid(example_images))

# Show noisy images: Add gaussian noise with the variance specified by
# variable 'noise_var' and mean = 0
noisy_examples = example_images + (noise_var ** 0.5) * torch.randn(example_images.shape)  # >> Your code goes here <<
imshow(torchvision.utils.make_grid(noisy_examples))

# Print labels
print(' '.join('%5s' % example_labels[j].item() for j in range(batch_size)))




epochs = 50
batchSize = 64
learningRate = 0.0001

generator = Generator.Generator()

discriminator = Discriminator.Discriminator()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Instantiate model and optimizer

model = Generator.Autoencoder(28 * 28).to(device)

print("Number of trainable parameters: {}".format(count_parameters(model)))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

tr_losses = np.zeros(num_epochs)
val_losses = np.zeros(num_epochs)

for epoch_nr in range(num_epochs):

    print("Epoch {}:".format(epoch_nr))

    # Train model
    running_loss = 0.0
    # >> Your code goes here <<
    for batch_data, _ in trainloader:
        # add noise (noisy version = input)
        batch_input = batch_data + (noise_var ** 0.5) * torch.randn(batch_data.shape)

        # Put data on device
        batch_data = batch_data.to(device)
        batch_input = batch_input.to(device)

        # Predict and get loss
        predictions = model(batch_input)
        loss = criterion(predictions, batch_data)  # batch_data = label here

        # Update model
        optimizer.zero_grad()  # re-initialize the gradient to zero
        loss.backward()
        optimizer.step()

        # Keep running statistics
        running_loss += loss.item()

    # Print results
    tr_loss = running_loss / len(trainloader.dataset)
    print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f}'.format(
        epoch_nr, running_loss / len(trainloader.dataset)))

    # Get validation results
    running_loss = 0
    # >> Your code goes here <<
    with torch.no_grad():
        for batch_data, _ in valloader:
            # add noise (noisy version = input)
            batch_input = batch_data + (noise_var ** 0.5) * torch.randn(batch_data.shape)

            # Put data on device
            batch_data = batch_data.to(device)
            batch_input = batch_input.to(device)

            # Predict and get loss
            predictions = model(batch_input)
            loss = criterion(predictions, batch_data)  # batch_data is the label here

            # Keep running statistics
            # running_loss += compute_run_acc(predictions, batch_data)#batch_data = label
            running_loss += criterion(predictions, batch_data)  # batch_data = label

    val_loss = running_loss / len(valloader.dataset)
    print('>> VALIDATION: Epoch {} | val_loss: {:.4f}'.format(epoch_nr, val_loss))

    tr_losses[epoch_nr] = tr_loss
    val_losses[epoch_nr] = val_loss

print('Training finished')


plt.figure()
plt.plot(tr_losses, label='Training')
plt.plot(val_losses, label='Validation')
plt.title('Results')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# Print some samples of dataset as a sanity check

# Get some random validation images
dataiter = iter(valloader)
example_images, _ = next(dataiter)

print(example_images.shape)

# Show images
imshow(torchvision.utils.make_grid(example_images))
noisy_data = example_images + (noise_var**0.5)*torch.randn_like(example_images)
imshow(torchvision.utils.make_grid(noisy_data))
preds = model(noisy_data.to(device))
imshow(torchvision.utils.make_grid(preds.cpu()))
# Print labels
print(' '.join('%5s' % example_labels[j].item() for j in range(batch_size)))

