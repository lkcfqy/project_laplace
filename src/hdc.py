# src/hdc.py
"""
Hyperdimensional Computing (HDC) Module
--------------------------------------
This module implements HDC operations including binding, bundling, and similarity.
It also includes a GridEncoder to encode 2D grids into hypervectors.
"""
import torch
import torch.nn.functional as F

class HDCSpace:
    def __init__(self, dim=10000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dim = dim
        self.device = device
        # Cache for item memory (random orthogonal vectors)
        self.memory = {}
        
    def get_random_vector(self, k=1):
        """Generate k random bipolar vectors {-1, 1}."""
        # Typically HDC uses -1, 1 for binding via element-wise mult (XOR alike)
        v = torch.randint(0, 2, (k, self.dim), device=self.device) * 2 - 1
        return v.float()

    def get_item_vector(self, key):
        """Retrieve or create a persistent vector for a specific key (e.g. 'Color_1')."""
        if key not in self.memory:
            self.memory[key] = self.get_random_vector(1).squeeze(0)
        return self.memory[key]
        
    def bind(self, v1, v2):
        """Element-wise multiplication (equivalent to XOR for bipolar)."""
        return v1 * v2
        
    def bundle(self, vectors):
        """Superposition (Element-wise addition). Clamped to stay bipolar-ish if needed, 
           but here we keep it continued for cosine sim."""
        # vectors: (N, dim)
        if len(vectors) == 0:
            return torch.zeros(self.dim, device=self.device)
        return torch.sum(vectors, dim=0)
    
    def permute(self, v, shifts=1):
        """Cyclic shift for sequence/position encoding."""
        return torch.roll(v, shifts, dims=-1)

    def similarity(self, v1, v2):
        """Cosine similarity."""
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

class GridEncoder:
    def __init__(self, hdc_space):
        self.space = hdc_space
        
    def encode_grid(self, grid):
        """
        Encode a 2D grid into a single vector.
        Formula: Sum( Position_Vector(r,c) * Color_Vector(val) )
        """
        vectors = []
        rows = len(grid)
        if rows == 0: return self.space.bundle([])
        cols = len(grid[0])
        
        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                
                # Create position-aware binding
                # Method: V_pos = Permute(V_axis_r, r) * Permute(V_axis_c, c)
                # Simple alternative: V_pos = Permute(V_base, r * 100 + c) for unique pos
                v_pos = self.space.permute(self.space.get_item_vector("pos_base"), r * 30 + c)
                
                v_val = self.space.get_item_vector(f"color_{val}")
                
                bound = self.space.bind(v_pos, v_val)
                vectors.append(bound)
                
        return self.space.bundle(torch.stack(vectors))
    
    def compute_diff(self, v1, v2):
        """
        Compute V_diff = V2 - V1 (approximate).
        In bipolar space, subtraction isn't standard binding, but for continuous representation 
        element-wise subtraction indicates 'what changed'.
        A better check for transformation is: does V1 + Shift == V2?
        For now, we return simple difference for distance checks.
        """
        return v2 - v1
