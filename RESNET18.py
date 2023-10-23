#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch import optim
import numpy as np
import csv
import random
from training_utils import LRDecayWithPatience, EarlyStopping

class RESNET18(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Define your custom architecture here
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8192, 256)  # Add fc1 layer
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)  # Add fc2 layer for classification
        
        # loss function for training
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # moves the entire model to the specified device (GPU or CPU).
        self.to(self.device)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        # Define the backbone architecture here
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        return x  # Features compatible with fc1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass using the backbone function
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)  # Classification layer (fc2)
        return x




    def extract_features(self, dataloader) -> torch.Tensor:
            features = []
            targets  = []
            bias_tgs = []
            #disables dropout and batch normalization layers
            self.eval()
            #no gradient information is stored
            with torch.no_grad():
                for inputs, labels, bias_labels in dataloader:
                    inputs, labels, bias_labels = inputs.to(self.device), labels.to(self.device), bias_labels.to(self.device)
                    for x, l, b in zip(inputs, labels, bias_labels):
                        #extract features from the input tensor
                        features.append(self.backbone(x).flatten().cpu().numpy())
                        targets.append(l.cpu())
                        bias_tgs.append(b.cpu())
            return np.array(features), np.array(targets), np.array(bias_tgs)
        
    def predict_from(self, dataloader, max_cont=0.2):
        predictions = []
        targets = []
        self.eval()
        with torch.no_grad():
            for inputs, labels, bias_labels in dataloader:
                inputs, labels, bias_labels = inputs.to(self.device), labels.to(self.device), bias_labels.to(self.device)
                for x, l, b in zip(inputs, labels, bias_labels):
                    #extract features from the input tensor
                    predictions.append(self(x).max(1).indices.cpu())
                    targets.append(l.cpu())

        predictions = np.array(predictions)
        targets = np.array(targets)
        error_rate = np.zeros((len(np.unique(targets))))
        for i in range(0,10):      
            targets_i = np.where(targets==i)[0]
            error_rate[i] = min(max_cont, np.count_nonzero(predictions[targets_i]!=i) / len(targets_i))
            if error_rate[i] == 0:
                error_rate[i]=0.01
        return np.array(predictions), np.array(targets), error_rate  


    def train_model_step0(
            self,
            train_loader,
            val_loader,
            test_loader,
            learning_rate=0.01,
            num_epochs=20):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, amsgrad=False)

        # loop that iterates over a specified number of training epochs
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            # loop that iterates over batches
            self.train()
            with torch.enable_grad():
                for inputs, labels, _ in train_loader:
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    #clears the gradients of the model's parameters
                    optimizer.zero_grad()
                    #computes the model's predictions
                    outputs = self(inputs)

                    #This computes the loss between the model's predictions and the actual labels
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    #updates the model's parameters using the optimizer, applying gradient descent.
                    optimizer.step()
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                #average training loss for that epoch
            average_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}/{num_epochs}\n\t -- Train Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            #computes the validation and test losses and calculates the accuracy of the model's predictions.
            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self(inputs)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            average_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            print(f"\t -- Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

            self.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    labels = labels.type(torch.LongTensor)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self(inputs)
                    loss = self.loss_fn(outputs, labels)
                    test_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels
                                ).sum().item()

            average_loss = test_loss / len(test_loader)
            accuracy = 100 * correct / total
            print(f"\t -- Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")


    def train_model_iter(
            self,
            train_loader,
            val_loader,
            test_loader,
            adecs,
            ground_truth,
            log_file_path,
            tr_accuracy_unsup,
            rand,
            learning_rate=0.01,
            num_epochs=2,
            bias_amount=0.95):

        # optimizer = optim.SGD(self.parameters(), lr=learning_rate)                                                                                                                 #faactor 0.9
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, amsgrad=False)
        # Define the ReduceLROnPlateau scheduler
        scheduler = LRDecayWithPatience(optimizer, verbose=True, min_lr=1e-9, threshold=1e-6, decay_factor=0.75, patience=5)
        early_stopping = EarlyStopping(patience=10, epsilon=1e-6)
        self.train()

        # Open a CSV file for logging
        with open(log_file_path, mode='w', newline='') as log_file:
            log_writer = csv.writer(log_file)
        # Write the header row to the CSV file
            log_writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"])
        # loop that iterates over a specified number of training epochs
            p_tensor = None
            for epoch in range(num_epochs):
                total_loss = 0.0
                correct_train = 0
                total_train = 0
                correct_biased=0
                correct_unbiased=0
                total_biased=0
                total_unbiased=0

                

                # loop that iterates over batches
                for inputs, labels, bias_labels in train_loader: 
                #vettore di weights di batchsize elementi
                    weights=[]
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    labels = labels.type(torch.LongTensor)
                    for i, l in enumerate(labels):
                        if rand is False:            
                            if ground_truth is False:
                                with torch.no_grad():
                                    if epoch==0:
                                        features = self.backbone(inputs[i]).cpu().numpy()
                                        p_tensor = adecs[l].predict(features)  
                                        p = int(p_tensor.item())
                                    weights.append((1-tr_accuracy_unsup[l]) if p == 1 else tr_accuracy_unsup[l])
                            else:
                                weights.append((1-bias_amount) if bias_labels[i] == 1 else bias_amount)
                        else:
                            values = [(1-bias_amount), bias_amount]
                            probabilities = [bias_amount, (1-bias_amount)]
                            # Generate the array
                            weights = random.choices(values, probabilities, k=len(labels))

                    
                    # Convert weights to a PyTorch tensor
                    weights_tensor = torch.tensor(weights)
                    weights_tensor=weights_tensor.to(self.device)
                    #clears the gradients of the model's parameters
                    optimizer.zero_grad()
                    #computes the model's predictions
                    outputs = self(inputs)
                    outputs=outputs.to(self.device)
                    labels=labels.to(self.device)
                    #This computes the loss between the model's predictions and the actual labels
                    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fn(outputs, labels)               
                    loss= loss*weights_tensor

                    # Make sure loss is a tensor
                    loss = loss.type(torch.FloatTensor)
                    loss.mean().backward()
                    #updates the model's parameters using the optimizer, applying gradient descent.
                    optimizer.step()
                    total_loss += loss.mean().item()
                    _, predicted_train = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted_train == labels).sum().item()

                    bias_predicted=predicted_train[bias_labels==1]
                    total_biased+=len(bias_predicted)
                    correct_biased+=(bias_predicted==labels[bias_labels==1]).sum().item()

                    unbias_predicted=predicted_train[bias_labels==-1]
                    total_unbiased+=len(unbias_predicted)
                    correct_unbiased+=(unbias_predicted==labels[bias_labels==-1]).sum().item()
                
                #average training loss for that epoch
                average_loss_train = total_loss / len(train_loader)
                accuracy_train = 100 * correct_train / total_train

                accuracy_train_biased=100*correct_biased/total_biased
                accuracy_train_unbiased=100*correct_unbiased/total_unbiased
            # Calculating training accuracy
                print(f"Epoch {epoch + 1}/{num_epochs}\n\t -- Loss: {average_loss_train:.4f}, Train Accuracy: {accuracy_train:.2f}%")
                print(f"\t\t Train Accuracy Biased: {accuracy_train_biased}%, Train Accuracy Unbiased: {accuracy_train_unbiased}%")

                self.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                correct_biased=0
                correct_unbiased=0
                total_biased=0
                total_unbiased=0
            
                #computes the validation and test losses and calculates the accuracy of the model's predictions.
                with torch.no_grad():
                    for inputs, labels, bias_labels in val_loader:
                        labels = labels.type(torch.LongTensor)
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self(inputs)
                        loss = self.loss_fn(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        bias_predicted=predicted[bias_labels==1]
                        total_biased+=len(bias_predicted)
                        correct_biased+=(bias_predicted==labels[bias_labels==1]).sum().item()

                        unbias_predicted=predicted[bias_labels==-1]
                        total_unbiased+=len(unbias_predicted)
                        correct_unbiased+=(unbias_predicted==labels[bias_labels==-1]).sum().item()

                average_loss_val = val_loss / len(val_loader)
                accuracy_val_biased=100*correct_biased/total_biased
                accuracy_val_unbiased=100*correct_unbiased/total_unbiased
                scheduler(average_loss_val)
                if early_stopping(average_loss_val, epoch):
                    print("EARLY STOPPING: no improvement for 10 epochs, stopping")
                    break
                accuracy_val = 100 * correct / total
                print(f"\t -- Validation Loss: {average_loss_val:.4f}, Accuracy: {accuracy_val:.2f}%")
                print(f"\t\t Valid Accuracy Biased: {accuracy_val_biased}%, Valid Accuracy Unbiased: {accuracy_val_unbiased}%")
                
                

                self.eval()
                test_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, labels, _ in test_loader:
                        labels = labels.type(torch.LongTensor)
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        outputs = self(inputs)
                        loss = self.loss_fn(outputs, labels)
                        test_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                average_loss_test = test_loss / len(test_loader)
                accuracy_test = 100 * correct / total
                print(f"\t -- Test Loss: {average_loss_test:.4f}, Accuracy: {accuracy_test:.2f}%")
                
                # Update the CSV file with test information
                # Store the data for this epoch in a list
                epoch_data = [epoch + 1, average_loss_train, accuracy_train, average_loss_val, accuracy_val, average_loss_test, accuracy_test]
            # Write the data for this epoch as a single row in the CSV file
                log_writer.writerow(epoch_data)

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

    @staticmethod
    def load_model(filepath):
        model = RESNET18()
        model.load_state_dict(torch.load(filepath))
        return model