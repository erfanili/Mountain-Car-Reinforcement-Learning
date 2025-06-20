import numpy as np

class TileCoder:
    def __init__(self, position_range, velocity_range, num_tilings = 8, tiles_per_dim = 8):
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.position_min, self.position_max = position_range
        self.velocity_min, self.velocity_max = velocity_range
        
        self.position_scale = self.tiles_per_dim / (self.position_max - self.position_min)
        self.velocity_scale = self.tiles_per_dim / (self.velocity_max - self.velocity_min)
        
        self.offsets = np.linspace(0, 1, num_tilings, endpoint=False)
        
        self.tiles_per_tiling = self.tiles_per_dim * self.tiles_per_dim

        
    def get_features(self, position, velocity):
        features = []
        for i in range(self.num_tilings):
            offset = self.offsets[i]
            pos_idx = int((position - self.position_min) * self.position_scale + offset)
            vel_idx = int((velocity - self.velocity_min) * self.velocity_scale + offset)
            tile_index = (pos_idx % self.tiles_per_dim) + (vel_idx % self.tiles_per_dim) * self.tiles_per_dim
            index = i * self.tiles_per_tiling + tile_index
            features.append(index)
        return features

    def feature_vector_length(self):
        return self.num_tilings * self.tiles_per_tiling


        