import torch
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class GCNAnalysis:
    """Class for training and evaluating the GCN model."""
    def __init__(self, num_features, num_classes, learning_rate=0.01, weight_decay=5e-4):
        self.model = GCN(num_features, num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train(self, train_data, num_epochs=50):
        """Train the GCN model."""
        self.model.train()
        train_losses = []

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            out = self.model(train_data)
            loss = F.nll_loss(out, train_data.y)
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss.item())

        return train_losses

    def evaluate(self, test_data):
        """Evaluate the GCN model on the test set."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_data)
            pred = out.argmax(dim=1).detach().numpy()
            y_true = test_data.y.detach().numpy()

        # Calculate evaluation metrics
        acc = accuracy_score(y_true, pred)
        pre = precision_score(y_true, pred, pos_label=1)
        rec = recall_score(y_true, pred, pos_label=1)
        f1 = f1_score(y_true, pred, pos_label=1)

        return {"accuracy": acc, "precision": pre, "recall": rec, "f1": f1}

    def save_losses(self, train_losses, test_losses, filename):
        """Save training and testing losses to a CSV file."""
        loss_data = pd.DataFrame(
            {"Epoch": range(1, len(train_losses) + 1), "Training Loss": train_losses, "Testing Loss": test_losses}
        )
        loss_data.to_csv(filename, index=False)