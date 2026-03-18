import os
import cudf
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class PlantDALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, img_dir, csv_file, training=True):
        super(PlantDALIPipeline, self).__init__(batch_size, num_threads, device_id, seed=42)
        
        # 1. Load metadata using cuDF (GPU accelerated)
        df = cudf.read_csv(csv_file)
        self.file_paths = [os.path.join(img_dir, f) for f in df.iloc[:, 0].to_arrow().to_pylist()]
        self.labels = df.iloc[:, 1].to_arrow().to_pylist()
        
        # 2. DALI Readers and Decoders
        self.input = ops.readers.File(
            files=self.file_paths, 
            labels=self.labels, 
            random_shuffle=training, 
            name="Reader"
        )
        
        # Decode on GPU (using mixed device: CPU for reading, GPU for decoding)
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        
        # 3. GPU Augmentations (448px for BioCLIP)
        self.resizer = ops.RandomResizedCrop(
            device="gpu", 
            size=448, 
            random_area=[0.08, 1.0]
        )
        self.cpoint_flip = ops.Flip(device="gpu", vertical=0, horizontal=ops.random.CoinFlip(probability=0.5))
        
        # 4. Normalization (BioCLIP specific means/stds)
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

def get_dali_loader(csv_path, img_dir, batch_size=128, num_threads=4, device_id=0):
    pipe = PlantDALIPipeline(
        batch_size=batch_size, 
        num_threads=num_threads, 
        device_id=device_id, 
        img_dir=img_dir, 
        csv_file=csv_path
    )
    pipe.build()
    
    # DALIGenericIterator wraps the pipeline for PyTorch
    return DALIGenericIterator(
        [pipe], 
        ['data', 'label'], 
        size=pipe.epoch_size("Reader"),
        auto_reset=True
    )
