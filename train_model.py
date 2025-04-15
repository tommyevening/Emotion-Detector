import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Prosty model CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*10*10, 128)  # 48x48 -> 10x10 po 2x Conv+Pool
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Ścieżki do danych
    train_dir = 'assets/train'
    test_dir = 'assets/test'

    # Transformacje (dostosowane do FER-2013 i Twoich danych)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Jeśli zdjęcia są kolorowe, zamień na 1 kanał
        transforms.Resize((48, 48)),                  # FER-2013: 48x48 px
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Wczytanie danych
    print("Start wczytywania danych...")
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    print("Wczytano train_data")
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    print("Wczytano test_data")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    num_classes = len(train_data.classes)
    model = SimpleCNN(num_classes)

    # Optymalizator i funkcja straty
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Trening
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Ewaluacja na zbiorze testowym
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

    # Zapisz model
    torch.save(model.state_dict(), 'assets/model_emocje.pth')
    print("Model zapisany jako assets/model_emocje.pth")