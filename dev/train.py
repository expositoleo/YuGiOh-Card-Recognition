import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import CustomImageDataset

IMAGES_DIR = "data/yugioh_card_images"

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        #self.resnet = models.resnet152(pretrained=True)
        self.resnet = models.resnet101(pretrained=True)
        #self.resnet = models.resnet50(pretrained=True)

        #self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def forward_once(self, x):
        '''
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        #print(output.shape)
        #print(output)
        '''
        #begin = time()
        output = self.resnet(x)
        #print('Time for forward prop: ', time()-begin)

        return output
        

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3
    

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = (anchor - positive).pow(2).sum(1)  # Squared distance
        neg_dist = (anchor - negative).pow(2).sum(1)  # Squared distance
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, images_pos, images_neg, ids, neg_ids in dataloader:
            images = images.to(device)
            images_pos = images_pos.to(device)
            images_neg = images_neg.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output1, output2, output3 = model(images, images_pos, images_neg)
            
            # Compute loss
            loss = criterion(output1, output2, output3)
            
            print(f"output1 shape: {output1.shape}, output2 shape: {output2.shape}, output3 shape: {output3.shape}")
            print(f"loss: {loss.item()}")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    print("Training complete!")
    return model

model = SiameseNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for param in model.resnet.parameters():
    param.requires_grad = True

transform = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.ToTensor()
])
dataset = CustomImageDataset(image_dir=os.path.abspath(IMAGES_DIR), transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

train_model(
    model=model,
    dataloader=dataloader,
    criterion=TripletLoss(margin=1.0),
    optimizer=optimizer,
    num_epochs=10
)