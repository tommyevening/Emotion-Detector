import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()


        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #1 Nums of channels if grayscale = 1, if RGB then 3, etc.
        #2 How many different features the model should look for
        #3 size of mesh
        #4 how many steps should kernel should make

        #Looking for more complex and abstarct features
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)
        # 2,2 - takes maximum val from matrix 2x2

        # Pierwsza warstwa w pełni połączona
        self.fc1 = nn.Linear(64*10*10, 128)
        #1 6400-dimension features represenattion 64-nums of out_channes from last convLayer 10x10 pooling
        #2 best effort 128

        #turns of 25% of neurons
        self.dropout = nn.Dropout(0.25)
        # 25% best offort

        self.fc2 = nn.Linear(128, num_classes)

        # f. activation
        self.relu = nn.ReLU() # [0, max]


    #Controll stream of data
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Spłaszczanie tensora do wektora
        x = x.view(x.size(0), -1)

        # First layer + ReLu
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        #Out layer
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    train_dir = 'assets/train'
    test_dir = 'assets/test'

    # Image transformation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  #if RGB transform to 1 channel
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # MIN output (0 - 0.5) / 0.5 = -0.5 / 0.5 = -1
        # MAX output (1 - 0.5) / 0.5 = 0.5 / 0.5 = 1
    ])

    # Load data
    print("Start wczytywania danych...")
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    print("Wczytano train_data")
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    print("Wczytano test_data")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    num_classes = len(train_data.classes)
    model = SimpleCNN(num_classes)

    # Optymalizator & loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # modifies weights during learning to minimalize loss | model.parameters() -> returns list of weight and bias | learning rate
    criterion = nn.CrossEntropyLoss() #How good my model is

    # Trening
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad() #grad = 0
            outputs = model(images) #model output
            loss = criterion(outputs, labels)
            loss.backward() #grad
            optimizer.step() #update weights
            running_loss += loss.item() #loss sum
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}") #avarage loss for training set



    # Eval on test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test accuracy: {100 * correct / total:.2f}%")


    #Save
    torch.save(model.state_dict(), 'assets/model_emocje.pth')
    print("Model weights saved as assets/model_emocje.pth")