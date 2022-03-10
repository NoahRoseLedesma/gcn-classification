import torch
from torch import nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report
import wandb

class GCN(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, config, dataset):
        super(GCN, self).__init__()
        self.dataset = dataset
        # Save the hyperparameters
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.weight_decay = config['weight_decay']
        self.dropout = config['dropout']
        self.hidden_dim = config['hidden_dim']

        # Put the dataset on the GPU
        self.dataset = self.dataset.to(self.device)

        # First convolutional layer
        self.conv1 = GCNConv(self.dataset.num_node_features, self.hidden_dim)
        self.conv1_activation = nn.ReLU()

        # Dropout
        self.dropout1 = nn.Dropout(self.dropout)

        # Second convolutional layer
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv2_activation = nn.ReLU()

        # Dropout
        self.dropout2 = nn.Dropout(self.dropout)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.dataset.num_classes)
        self.fc_activation = nn.Softmax(dim=1)

        self.to(self.device)

    def forward(self, dataset):
        x, edge_index = dataset.x, dataset.edge_index

        # First convolutional layer
        x = self.conv1_activation(self.conv1(x, edge_index))

        # Dropout
        x = self.dropout1(x)

        # Second convolutional layer
        x = self.conv2_activation(self.conv2(x, edge_index))

        # Dropout
        x = self.dropout2(x)

        # Fully connected layer
        x = self.fc_activation(self.fc(x))

        return x
        

    
    def train(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        wandb.watch(self)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            # Forward pass
            output = self.forward(self.dataset)[self.dataset.train_mask]
            y = self.dataset.y[self.dataset.train_mask]
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, pred = output.max(dim=1)
            correct = pred.eq(y).sum().item()

            if epoch % 10 == 0:
                print('Epoch: {:03d}, Loss: {:.4f}, Accuracy: {:.4f}'.format(
                    epoch, loss.item(), correct / len(y)))
            
            # Log to wandb
            wandb.log({'loss': loss.item(), 'accuracy': correct / len(y)})

    
    def evaluate(self):
        # Compute accuracy
        output = self.forward(self.dataset)[self.dataset.test_mask]
        y = self.dataset.y[self.dataset.test_mask]
        _, pred = output.max(dim=1)
        print(classification_report(y.cpu(), pred.cpu()))
        # Log the evaluation to wandb
        wandb.log({'evaluation': classification_report(y.cpu(), pred.cpu(), output_dict=True)})