import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import config
from hdc import HDCSpace, GridEncoder

class HDCDataset(Dataset):
    def __init__(self, trace_file, hdc_space):
        self.data = []
        self.hdc = hdc_space
        self.encoder = GridEncoder(hdc_space)
        self.trace_file = trace_file
        self.load_data()
        
    def load_data(self):
        if not os.path.exists(self.trace_file):
             return
             
        if not os.path.exists(self.trace_file):
             return
             
        with open(self.trace_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    state_grid = entry['state_grid']
                    target_grid = entry['target_grid']
                    label = float(entry['label'])
                    
                    state_vec = self.encoder.encode_grid(state_grid)
                    target_vec = self.encoder.encode_grid(target_grid)
                    
                    self.data.append((state_vec, target_vec, torch.tensor(label, dtype=torch.float32)))
                except Exception as e: 
                    # print(f"Skipping bad line: {e}")
                    continue
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # return state_vec, target_vec, label
        return self.data[idx]

def create_dataloader(trace_file, batch_size=32):
    hdc = HDCSpace()
    dataset = HDCDataset(trace_file, hdc)
    # Check if empty
    if len(dataset) == 0: return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
