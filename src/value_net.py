import torch
import torch.nn as nn
import torch.optim as optim

class HDCValueNet(nn.Module):
    def __init__(self, input_dim=20000, hidden_dim=512): # Input is 2 * hdc_dim (concatenated or diff)
        super(HDCValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Probability of this state leading to solution (or similarity score)
        )
        
    def forward(self, state_vec, target_vec):
        # We can concatenate [state, target] or use [state, target, state-target]
        # For simplicity: Concatenate
        x = torch.cat([state_vec, target_vec], dim=-1)
        return self.net(x)

def train_value_net(model, dataloader, epochs=5, device='cuda'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for state_vec, target_vec, label in dataloader:
            state_vec, target_vec, label = state_vec.to(device), target_vec.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(state_vec, target_vec)
            loss = criterion(output, label.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss {total_loss / len(dataloader)}")
    
    return model
