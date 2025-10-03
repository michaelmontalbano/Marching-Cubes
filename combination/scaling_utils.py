import numpy as np

class UnifiedScaler:
    def __init__(self, stats_path='global_mins3.npy', max_path='global_maxs3.npy'):
        """Initialize with saved training statistics"""
        self.global_mins = np.load(stats_path)
        self.global_maxs = np.load(max_path)
        
        # Correct channel mapping with MESH at indices 1 and 7
        self.channel_mapping = {
            0: 'MESH_Max_60min',
            1: 'MESH',  # First MESH - skipped in normalization
            2: 'HeightCompositeReflectivity',
            3: 'EchoTop_50',
            4: 'PrecipRate',
            5: 'Reflectivity_0C',
            6: 'Reflectivity_-20C',
            7: 'MESH'  # Second MESH - this one gets normalized
        }
        
    def scale_data(self, data, channel_idx):
        """Scale data exactly as in training"""
        # Skip channel 1 (first MESH) as in training
        if channel_idx == 1:
            return data.astype('float32')
            
        if channel_idx < len(self.global_mins):
            min_val = self.global_mins[channel_idx]
            max_val = self.global_maxs[channel_idx]
            
            # Apply exact same normalization as training
            scaled = (data - min_val) / (max_val - min_val + 1e-5)
            
            # Apply same floor value as training
            scaled = np.where(scaled <= 0, 1e-5, scaled)
            
            return scaled.astype('float32')
        else:
            raise ValueError(f"Channel index {channel_idx} exceeds available statistics")