import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import numpy as np
import torch
from jetnet.datasets import JetNet
from jetnet.utils import to_image
from tqdm import tqdm  
import os

# GPU setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
# Print GPU details if CUDA is available
if torch.cuda.device_count() > 1:
    print(f"{torch.cuda.device_count()} GPUs availables")

# Directories setup
PROJECT_PATH = './JetNet-GPU_dataset'
DATA_PATH = f'{PROJECT_PATH}/Datasets/JetNet'
os.makedirs(DATA_PATH, exist_ok=True)

output_dir = f'{DATA_PATH}/Jet2Image_5types_170k'
os.makedirs(output_dir, exist_ok=True)

# DATASET configuration
data_args = {
    #"jet_type": ['g'],  # Puedes cambiarlo por ['g', 'q', 't', 'w', 'z']
    "jet_type": ['g', 'q', 't', 'w', 'z'],
    "data_dir": DATA_PATH,
    "particle_features": "all",
    "num_particles": 30,
    "jet_features": "all",
    "download": True
}

# Data load
particle_data, jet_data = JetNet.getData(**data_args)

# Images configuration

num_images = 170000
num_types = len(data_args["jet_type"])
im_size = 299
maxR = 0.4
cm = plt.cm.jet.copy()
cm.set_under(color="white")
plt.rcParams.update({"font.size": 16})

# Optimized processing
type_indices = {jet_type: JetNet.JET_TYPES.index(jet_type) for jet_type in data_args["jet_type"]}
type_numbers = {jet_type: idx for idx, jet_type in enumerate(data_args["jet_type"])}
batch_size = 500  # Adjust according to available memory


def generate_images_batch(particle_batch, jet_type, type_num, start_idx):
    
    for i, particles in enumerate(tqdm(particle_batch, desc=f"Generating {jet_type} (type{type_num}) {start_idx}-{start_idx + len(particle_batch) - 1}")):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            to_image(particles, im_size, maxR=maxR),
            cmap=cm,
            interpolation="nearest",
            vmin=1e-8,
            extent=[-maxR, maxR, -maxR, maxR],
            vmax=0.05,
        )
        ax.axis('off')

        filename = f"{output_dir}/jet2image_{jet_type}_type{type_num}_num{start_idx + i}.png"
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)



for jet_type in data_args["jet_type"]:
    type_idx = type_indices[jet_type]
    type_num = type_numbers[jet_type]

    type_selector = jet_data[:, 0] == type_idx
    selected_data = particle_data[type_selector][:num_images]
    
    # Batch processing for better memory handling
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch = selected_data[batch_start:batch_end]
        
        # Move data to GPU if possible (for pre-processing)
        batch_tensor = torch.tensor(batch, device=device)
        
        # Generate images while maintaining original quality
        generate_images_batch(batch_tensor.cpu().numpy(), jet_type, batch_start)

print("¡Images Generated!")