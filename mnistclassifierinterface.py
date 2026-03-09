# ================================
# 1. Import libraries
# ================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC, abstractmethod

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================================
# 2. Data preparation
# ================================

# Transform images to tensors for PyTorch models
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform_train)
test_dataset  = datasets.MNIST(root="data", train=False, download=True, transform=transform_test)

# DataLoader for NN and CNN
train_subset, val_subset = random_split(train_dataset, [54000, 6000],
                               generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_subset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# For RandomForest — convert to numpy arrays
X_train_rf = train_dataset.data.view(-1, 28*28).numpy().astype(np.float32) / 255.0
y_train_rf = train_dataset.targets.numpy().astype(np.int64)
X_test_rf  = test_dataset.data.view(-1, 28*28).numpy().astype(np.float32) / 255.0
y_test_rf  = test_dataset.targets.numpy().astype(np.int64)

# ================================
# 3. MnistClassifierInterface
# ================================
class MnistClassifierInterface(ABC):
    @abstractmethod
    def fit(self, train_data, train_labels=None):
        pass

    @abstractmethod
    def predict(self, x):
        pass

# ================================
# 4. Random Forest Model with Randomized Search
# ================================

# Random search parameters
param_dist = {
    'n_estimators': [150, 200],
    'max_depth': [None, 30],
    'max_features': ['sqrt'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True],
    'criterion': ['gini', 'entropy']
}

# Model
rf = RandomForestClassifier(random_state=42)

# Using RandomizedSearchCV is faster because it only tests n_iter combinations
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_dist,
                                   n_iter=20,  # 20 random combinations
                                   cv=3,
                                   scoring='accuracy',
                                   verbose=1,
                                   n_jobs=-1,
                                   random_state=42)

# Model training
random_search.fit(X_train_rf, y_train_rf)

# Result
print("The best parameters:", random_search.best_params_)
print("Best accuracy (CV):", random_search.best_score_)

# Test set score
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_rf)
test_acc = accuracy_score(y_test_rf, y_pred)
print(f"Accuracy on the test: {test_acc:.4f}")

# ================================
# Random Forest Implementation
# ================================

class RandomForestMnist(MnistClassifierInterface):
    def __init__(self, **rf_params):
        self.model = RandomForestClassifier(
            n_estimators=150, min_samples_split=2, min_samples_leaf=1,
            max_features='sqrt', max_depth= None,
            criterion='gini', bootstrap=True,
            random_state=42
        )

    def fit(self, train_data, train_labels=None, **kwargs):
        if train_labels is None:
            raise ValueError("train_labels required for RandomForestMnist.fit")
        self.model.fit(train_data, train_labels)

    def predict(self, x, **kwargs):
        return self.model.predict(x)

# ================================
# 5. Base class for trainable PyTorch models with training, validation, and early stopping
# ================================

class TrainableNN(nn.Module, MnistClassifierInterface):
    def __init__(self):
        super().__init__()

    def fit(self, train_data, val_data=None, epochs=50, lr=0.001, device="cpu", early_stopping_patience=5):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_model_wts = copy.deepcopy(self.state_dict())
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            self.train()
            running_loss = 0.0
            num_batches = 0
            for images, labels in train_data:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num_batches += 1
            train_loss = running_loss / max(1, num_batches)

            # Validation
            if val_data is not None:
                self.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for images, labels in val_data:
                        images, labels = images.to(device), labels.to(device)
                        outputs = self(images)
                        val_loss += loss_fn(outputs, labels).item()
                        val_batches += 1
                val_loss /= max(1, val_batches)

                print(f"Epoch [{epoch+1}/{epochs}] — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

                # Save best
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch [{epoch+1}/{epochs}] — Train Loss: {train_loss:.4f}")

        # Restore best
        self.load_state_dict(best_model_wts)
        print(f"\nRestored best model (Val Loss: {best_loss:.4f})")
        checkpoint_path = f"{self.__class__.__name__}_best.pth"   # FeedForwardNN_best.pth або CNNClassifierKerasStyle_best.pth
        torch.save(best_model_wts, checkpoint_path)
        print(f"Saved to {checkpoint_path}")

    def predict(self, x, device="cpu"):
        # Make predictions on a batch or full dataset
        self.eval()
        self.to(device)
        x = x.to(device)
        with torch.no_grad():
            outputs = self(x)
            return torch.argmax(outputs, dim=1)

# ================================
# 6. Feed-Forward Neural Network
# ================================

class FeedForwardNN(TrainableNN):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

# ================================
# 7. Convolutional Neural Network
# ================================

class CNNClassifierKerasStyle(TrainableNN):
    def __init__(self):
        super().__init__()
        # --- Convolutional layers ---
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # --- Pooling ---
        self.pool = nn.MaxPool2d(2, 2)

        # --- Dropout ---
        self.dropout_conv = nn.Dropout(0.25)
        self.dropout_fc = nn.Dropout(0.5)

        # --- Fully connected ---
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)

# ==================
# 8. MnistClassifier
# ==================

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.model = RandomForestMnist()
        elif algorithm == "nn":
            self.model = FeedForwardNN()
        elif algorithm == "cnn":
            self.model = CNNClassifierKerasStyle()
        else:
            raise ValueError("Unknown algorithm")

    def fit(self, train_data, train_labels=None, **kwargs):
      if isinstance(self.model, TrainableNN):
        self.model.fit(train_data, **kwargs)
      else:
        self.model.fit(train_data, train_labels)

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)

# ================================
# 9. Evaluation Function
# ================================

def evaluate(classifier: MnistClassifier, loader, device="cpu"):
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device)
        preds = classifier.predict(images, device=device)
        correct += (preds == labels.to(device)).sum().item()
        total += labels.size(0)
    return correct / total

# ================================
# 10. Example Usage
# ================================

# ---- Random Forest ----
rf = RandomForestMnist()
rf.fit(X_train_rf, y_train_rf)

rf_preds = rf.predict(X_test_rf)
rf_acc = accuracy_score(y_test_rf, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_rf, rf_preds))

"""Random Forest serves as a baseline classical ML model.
The model performs very well with an overall accuracy of 97% on the test set.

The F1-scores for all classes range between 0.96 and 0.99, indicating balanced precision and recall across digits.

Some digits are easier to recognize, e.g., 0, 1, and 6 have high recall values (>0.98), meaning the model correctly identifies almost all instances of these digits.

Some digits are slightly more challenging, e.g., 9 has a recall of 0.95, showing that it is occasionally misclassified.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Feed-Forward Neural Network ---

nn_model = MnistClassifier("nn")
nn_model.fit(train_loader, val_data=val_loader,
             epochs=30, lr=0.001, device=device, early_stopping_patience=7)
nn_acc = evaluate(nn_model, test_loader, device=device)
print(f"Feed-Forward NN Accuracy: {nn_acc:.4f}")

"""Training loss decreased from 0.5198 to 0.1089, indicating good learning of the patterns in the training set.

Validation loss reached its minimum at 0.0589, showing the model generalizes well to unseen data without significant overfitting.

The final accuracy on the test set is 99.02%.
"""

# --- Convolutional Neural Network ---

cnn_model = MnistClassifier("cnn")
cnn_model.fit(train_loader, val_data=val_loader,
              epochs=100, lr=0.0008, device=device, early_stopping_patience=10)
cnn_acc = evaluate(cnn_model, test_loader, device=device)
print(f"CNN Accuracy: {cnn_acc:.4f}")
