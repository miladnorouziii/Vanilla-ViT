import Modules.DataHandler as data
from Modules.ViT import ViT
from colorama import init
import os 
import torch
from timeit import default_timer as timer
from tqdm.auto import tqdm

class Main():

    device = "cpu"
    configs = {"epochs": None,
        "batchSize": 20
    }


    def colorText(self, text, color):
        init()
        colorCode = ""
        if color == "g":
            colorCode = "\033[32m"
        elif color == "r":
            colorCode = "\033[31m"
        elif color == "y":
            colorCode = "\033[33m"
        elif color == "c":
            colorCode = "\033[36m"
        elif color == "m":
            colorCode = "\033[35m"
        return f"{colorCode}{text}\033[0m"
    

    def checkHardware(self):
        print(self.colorText("Checking your hardware ...\n", "y"))
        try:
            os.system('nvidia-smi')
        except Exception as e:
            print(self.colorText(f"Error -> {e}\n", "r"))
        if torch.cuda.is_available():
            print(self.colorText("\nCUDA is available.", "g"))
            numberOfGpus = torch.cuda.device_count()
            print(self.colorText(f"Number of available GPUs: {numberOfGpus}", "g"))
            for i in range (numberOfGpus):
                gpuProperties = torch.cuda.get_device_properties(i)
                print(self.colorText(f"GPU{i}: {gpuProperties.name}, (CUDA cores: {gpuProperties.multi_processor_count})", "g"))
            self.device = torch.device("cuda")
        else:
            print(self.colorText("OOps! your GPU doesn't support required CUDA version. Running on CPU ...\n", "r"))
            self.device = torch.device("cpu")

    
    def getUserParams(self):
        self.configs["epochs"] = int(input("\n-> Enter iteration number: "))
        self.configs["batchSize"] = int(input("-> Enter batch size:(default 20): "))

    
    def accuracyFunc(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100 
        return acc
    

    def startNN(self):
        self.checkHardware()
        self.getUserParams()
        trainLoader, testLoader = data.getData(self.configs["batchSize"])
        vit = ViT(num_classes=10).to(self.device)
        optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=3e-3, # Base LR from Table 3 for ViT-* ImageNet-1k
                             betas=(0.9, 0.999), # default values but also mentioned in ViT paper section 4.1 (Training & Fine-tuning)
                             weight_decay=0.3) # from the ViT paper section 4.1 (Training & Fine-tuning) and Table 3 for ViT-* ImageNet-1k

# Setup the loss function for multi-class classification
        criterion = torch.nn.CrossEntropyLoss()
        modelStartTime = timer()
        for epoch in tqdm(range(self.configs['epochs'])):
            train_loss, train_acc = 0, 0
            for batch, (X, y) in enumerate(trainLoader):
                X, y = X.to(self.device), y.to(self.device)
                y_pred = vit(X)
                loss = criterion(y_pred, y)
                train_loss += loss
                train_acc += self.accuracyFunc(y_true=y, y_pred=y_pred.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss /= len(trainLoader)
            train_acc /= len(trainLoader)
            print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
            vit.eval() 
            with torch.inference_mode():
                test_loss, test_acc = 0, 0
                for batch, (inputs, labels) in enumerate(testLoader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = vit(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss
                    test_acc += self.accuracyFunc(y_true=labels, y_pred=outputs.argmax(dim=1))
                test_loss /= len(testLoader)
                test_acc /= len(testLoader)
                print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%")
        modelEndTime = timer()
        totalTime = modelEndTime - modelStartTime
        print(f"Train time on {self.device}: {totalTime:.3f} seconds")

if __name__ == "__main__":
    main = Main()
    main.startNN()