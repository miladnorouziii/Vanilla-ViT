#import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), 
    transforms.ToTensor() 
])


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

def getData(batchSize):
    trainData = datasets.CIFAR10(root="Dataset", train=True,
        download=True, transform=train_transforms
    )
    testData = datasets.CIFAR10(root="Dataset", train=False,
        download=True, transform=test_transforms
    )
    trainDataloader = DataLoader(trainData,
        batch_size=batchSize,  
        shuffle=True
    )
    testDataloader = DataLoader(testData,
        batch_size=batchSize,
        shuffle=False
    )
    return trainDataloader, testDataloader