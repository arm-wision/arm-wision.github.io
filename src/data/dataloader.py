import os
# Bypass NVML errors on WSL
os.environ["DALI_DONT_USE_NVML"] = "1"
os.environ["DALI_NVML_FORCE_NO_PCIE_GEN"] = "1"
import cudf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np

class PlantDALIPipeline(Pipeline):
    """
    NVIDIA DALI Pipeline for GPU-accelerated image loading and augmentation.
    Decodes JPEGs on the GPU and performs all augmentations (resize, flip, normalize)
    directly in VRAM to eliminate CPU bottlenecks.
    """
    def __init__(self, batch_size, num_threads, device_id, file_paths, labels, training=True):
        """
        Args:
            batch_size (int): Images per batch.
            num_threads (int): CPU threads for JPEG header decoding.
            device_id (int): GPU index.
            file_paths (list): List of absolute image paths.
            labels (list): List of integer species labels.
            training (bool): Whether to enable random augmentations.
        """
        super(PlantDALIPipeline, self).__init__(
            batch_size, 
            num_threads, 
            device_id, 
            seed=42, 
            prefetch_queue_depth=4 # Keeps the GPU fed by pre-loading 4 batches
        )
        
        self.file_paths = file_paths
        self.labels = np.array(labels, dtype=np.int32)
        
        # 1. Readers: Efficiently pull data from disk
        self.input = ops.readers.File(
            files=self.file_paths, 
            labels=list(self.labels), 
            random_shuffle=training, 
            name="Reader",
            num_shards=1,
            shard_id=0,
            pad_last_batch=True
        )
        
        # 2. Decoders: Standard CPU decoding (Safest for all driver environments)
        # Update to mixed decoding (CPU parsing + GPU bitstream decoding)
        self.decode = ops.decoders.Image(
            device="mixed",
            output_type=types.RGB,
            device_memory_padding=21101592, # mem headroom for lrg jpegs 
            host_memory_padding=8388608,
            # device="cpu"
        )
        
        # 3. Spatial Augmentations
        self.training = training
        if self.training:
            # Training: Inception-style random cropping
            self.resizer = ops.RandomResizedCrop(device="gpu", size=448, random_area=[0.08, 1.0])
            # self.resizer = ops.RandomResizedCrop(device="cpu", size=448, random_area=[0.08, 1.0])
            # Random horizontal flip generator
            self.coin = ops.random.CoinFlip(probability=0.5)
        else:
            # Validation: Standard center resize (fixed size to ensure uniform batch shapes)
            self.resizer = ops.Resize(device="gpu", size=(448, 448))
            # self.resizer = ops.Resize(device="cpu", size=(448, 448))
        
        self.flip = ops.Flip(device="gpu", vertical=0)
        # self.flip = ops.Flip(device="cpu", vertical=0)
        
        # 4. Normalization: Standardized means/stds for BioCLIP/DINOv2
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            # device="cpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
        )

    def define_graph(self):
        """ Defines the DALI execution graph. """
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.resizer(images)
        
        # Apply random horizontal flip during training
        if self.training:
            images = self.flip(images, horizontal=self.coin())
        else:
            images = self.flip(images, horizontal=0)
            
        output = self.normalize(images)
        return output.gpu(), labels.gpu() # output, labels.gpu()

def get_dali_loaders(csv_path, img_dir, batch_size=128, val_split=0.1, num_threads=4, device_id=0, sampling_mode='natural'):
    """
    Constructs high-performance training and validation DALI iterators.
    Supports various resampling strategies to handle the long-tail distribution.

    Args:
        sampling_mode (str): 
            'natural'  - Direct distribution from the CSV.
            'sqrt'     - Square-root resampling (P prop to 1/sqrt(N)). Best for Phase 2.
            'balanced' - Full class-aware balancing (P prop to 1/N).
    """
    # 1. Metadata Ingestion
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = cudf.read_csv(csv_path, sep=';')
    
    img_names = df['image_name'].to_arrow().to_pylist()
    species_ids = df['species_id'].to_arrow().to_pylist()
    
    # Standardize label encoding to 0...N-1
    unique_species = sorted(list(set(species_ids)))
    species_to_idx = {s: i for i, s in enumerate(unique_species)}
    all_labels = [species_to_idx[s] for s in species_ids]
    
    all_paths = [
        os.path.join(img_dir, str(sid), fname) 
        for fname, sid in zip(img_names, species_ids)
    ]
    
    # 2. Train/Val Splitting
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

    # 3. Long-Tail Calibration (Resampling)
    # This logic creates a weighted distribution where rare species appear more often.
    if sampling_mode in ['sqrt', 'balanced']:
        print(f"[Dataloader] Applying {sampling_mode} resampling...")
        
        # Calculate class frequencies in the training pool
        label_counts = {}
        for l in train_labels:
            label_counts[l] = label_counts.get(l, 0) + 1
            
        # Define sampling probability weights
        if sampling_mode == 'sqrt':
            class_weights = {l: 1.0 / np.sqrt(count) for l, count in label_counts.items()}
        else: # 'balanced'
            class_weights = {l: 1.0 / count for l, count in label_counts.items()}
            
        sample_weights = np.array([class_weights[l] for l in train_labels])
        sample_weights /= sample_weights.sum() # Normalize to sum to 1.0
        
        # Create a new, balanced list by sampling with replacement
        resampled_indices = np.random.choice(
            len(train_paths), 
            size=len(train_paths), 
            replace=True, 
            p=sample_weights
        )
        
        train_paths = [train_paths[i] for i in resampled_indices]
        train_labels = [train_labels[i] for i in resampled_indices]
        print(f"[Dataloader] Resampling complete. Effective samples: {len(train_paths):,}")
    
    # 4. Build Pipelines
    train_pipe = PlantDALIPipeline(
        batch_size, num_threads, device_id, train_paths, train_labels, training=True
    )
    val_pipe = PlantDALIPipeline(
        batch_size, num_threads, device_id, val_paths, val_labels, training=False
    )
    
    train_pipe.build()
    val_pipe.build()
    
    train_loader = DALIGenericIterator(
            [train_pipe], ['data', 'label'], reader_name="Reader", auto_reset=True
    )
    val_loader = DALIGenericIterator(
            [val_pipe], ['data', 'label'], reader_name="Reader", auto_reset=True
    )
    
    return train_loader, val_loader, len(unique_species)
