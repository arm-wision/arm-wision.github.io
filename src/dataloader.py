import os
import cudf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class PlantDALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, file_paths, labels, training=True):
        # Increase prefetch_queue_depth to keep the GPU fed
        super(PlantDALIPipeline, self).__init__(
            batch_size, 
            num_threads, 
            device_id, 
            seed=42, 
            prefetch_queue_depth=4 # Prefetch 4 batches ahead
        )
        
        self.file_paths = file_paths
        self.labels = labels
        
        # 1. DALI Readers and Decoders
        self.input = ops.readers.File(
            files=self.file_paths, 
            labels=self.labels, 
            random_shuffle=training, 
            name="Reader",
            # Parallelize file reading
            num_shards=1,
            shard_id=0,
            pad_last_batch=True
        )
        
        # Mixed device: CPU reads/decodes headers, GPU decodes pixels
        # Use device_memory_padding to avoid reallocations for different image sizes
        self.decode = ops.decoders.Image(
            device="mixed", 
            output_type=types.RGB,
            device_memory_padding=21102592, # ~20MB padding for 4090
            host_memory_padding=8388608     # ~8MB padding
        )
        
        # 2. GPU Augmentations
        if training:
            self.resizer = ops.RandomResizedCrop(device="gpu", size=448, random_area=[0.08, 1.0])
        else:
            self.resizer = ops.Resize(device="gpu", resize_shorter=448)
            
        self.cpoint_flip = ops.Flip(device="gpu", vertical=0, horizontal=ops.random.CoinFlip(probability=0.5) if training else 0)
        
        # 3. Normalization (Fully on GPU)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
        )

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.resizer(images)
        images = self.cpoint_flip(images)
        output = self.normalize(images)
        return output, labels.gpu()

def get_dali_loaders(csv_path, img_dir, batch_size=128, val_split=0.1, num_threads=8, device_id=0):
    # ... (rest of the metadata loading logic remains the same)

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.resizer(images)
        images = self.cpoint_flip(images)
        output = self.normalize(images)
        return output, labels.gpu()

def get_dali_loaders(csv_path, img_dir, batch_size=128, val_split=0.1, num_threads=4, device_id=0):
    # 1. Load and Process Metadata
    df = cudf.read_csv(csv_path, sep=';')
    
    img_names = df.iloc[:, 0].to_arrow().to_pylist()
    species_ids = df.iloc[:, 2].to_arrow().to_pylist()
    
    # Consistent Label Encoding
    unique_species = sorted(list(set(species_ids)))
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    all_labels = [species_to_idx[s] for s in species_ids]
    
    all_paths = [
        os.path.join(img_dir, str(sid), fname) 
        for fname, sid in zip(img_names, species_ids)
    ]
    
    # 2. Split Data
    import numpy as np
    indices = np.arange(len(all_paths))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_size = int(len(all_paths) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_paths = [all_paths[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    val_paths = [all_paths[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # 3. Build Pipelines
    train_pipe = PlantDALIPipeline(batch_size, num_threads, device_id, train_paths, train_labels, training=True)
    val_pipe = PlantDALIPipeline(batch_size, num_threads, device_id, val_paths, val_labels, training=False)
    
    train_pipe.build()
    val_pipe.build()
    
    train_loader = DALIGenericIterator([train_pipe], ['data', 'label'], size=len(train_paths), auto_reset=True)
    val_loader = DALIGenericIterator([val_pipe], ['data', 'label'], size=len(val_paths), auto_reset=True)
    
    return train_loader, val_loader, len(unique_species)
