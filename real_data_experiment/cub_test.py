import os
import shutil
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from datetime import datetime
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from functools import partial
from tests import (crossMMD2sampleUnpaired, safe_crossSSMMD2sample, 
                   TwoSampleMMDSquared, crossSSMMD2sample)
from utils import (RBFkernel, RBFkernel1, get_bootstrap_threshold, 
                   get_normal_threshold, get_spectral_threshold, 
                   GaussianVector, get_median_bw)
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA, KernelPCA
from torch.utils.data import TensorDataset, DataLoader
import torchvision

# Function to convert tensors to numpy arrays
def to_cpu_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif isinstance(obj, list):
        return [to_cpu_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_cpu_numpy(v) for k, v in obj.items()}
    else:
        return obj
    
# Function to format parameters for folder names
def format_embed_folder(hidden, out, do, bn, act):
    if isinstance(hidden, list):
        hidden_str = "[" + ",".join(map(str, hidden)) + "]"
    else:
        hidden_str = str(hidden)
    return f"hid_{hidden_str}_out_{out}_do_{do}_bn_{bn}_act_{act.__name__}"

def add_noise_to_embeddings(embeddings, noise_level=0.1):
    noise = torch.randn_like(embeddings) * noise_level
    return embeddings + noise

# Function to normalize a vector
def normalize_vector(vector):
    std = np.std(vector)
    if std ==0:
        return vector
    return vector / std

# MLP model definition
def make_mlp(input_dim, hidden_dims, output_dim, dropout=0.0, batchnorm=False, activation=nn.ReLU):
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode
bert_model.eval()

# Text preprocessing for BERT
def preprocess_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    return inputs

# Extract text embeddings using BERT and save them
# Process and save text embedding
def process_and_save_text_embedding_with_mlp(text_path, save_path, mlp_model):
    with open(text_path, 'r', encoding='ISO-8859-1') as file:
        text = file.read()

    inputs = preprocess_text(text)
    mlp_model.eval()  
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    reduced_embedding = mlp_model(cls_embedding)
    reduced_embedding_np = reduced_embedding.detach().numpy()  
    
    if os.path.exists(save_path):
        print(f"Embedding already exists for {save_path}. Skipping.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, reduced_embedding_np)
    
    print(f"Processing text: {text_path}")

# Pretrained ResNet50 model for image embedding
resnet152 = models.resnet152(pretrained=True)
resnet152_fc_removed = torch.nn.Sequential(*list(resnet152.children())[:-1])
resnet152.eval()

# Image preprocessing
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)


# Process and save image embedding
def process_and_save_image_embedding_with_mlp(image_path, save_path, mlp_model, input_dim=2048, hidden_dim=512, output_dim=128):
    image_tensor = preprocess_image(image_path)
    mlp_model.eval()  
    with torch.no_grad():
        embedding = resnet152_fc_removed(image_tensor)  
        embedding = embedding.view(1, -1)
    
    reduced_embedding = mlp_model(embedding)  
    reduced_embedding_np = reduced_embedding.squeeze().detach().numpy() 
    
    if os.path.exists(save_path):
        print(f"Embedding already exists for {save_path}. Skipping.")
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, reduced_embedding_np)

    print(f"Processing image: {image_path}")


# Generate embeddings for a single folder
def generate_embeddings_for_folder(folder, base_image_path, base_text_path, embed_image_path, embed_text_path):
    image_folder = os.path.join(base_image_path, folder)
    text_folder = os.path.join(base_text_path, folder)
    print(text_folder)
    for file_name in sorted(os.listdir(image_folder)):
        
        if file_name.endswith((".jpg", ".png")):
            image_path = os.path.join(image_folder, file_name)
            text_file_name = file_name.replace(".jpg", ".txt").replace(".png", ".txt")
            text_path = os.path.join(text_folder, text_file_name)
            
            if os.path.exists(text_path):
                # Save text embedding
                text_save_path = os.path.join(embed_text_path, folder, text_file_name.replace(".txt", ".npy"))
                process_and_save_text_embedding_with_mlp(text_path, text_save_path, mlp_text)
            if os.path.exists(image_path):
                # Save image embedding
                image_save_path = os.path.join(embed_image_path, folder, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))
                process_and_save_image_embedding_with_mlp(image_path, image_save_path, mlp_image)


# Combine image and text embeddings
def combine_embeddings(image_embeddings, text_embeddings):
    combined = [np.concatenate([img, txt]) for img, txt in zip(image_embeddings, text_embeddings)]
    return np.array(combined)  # Shape: (N, 2816)

def load_embeddings(folder_path):
    
    embeddings = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            embedding = np.load(file_path)
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]  # (1, D)
            elif embedding.ndim == 3 and embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)  # (N, 1, D) → (N, D)
            embeddings.append(embedding)
    if embeddings:
        return np.concatenate(embeddings, axis=0)
    else:
        return np.array([])
    
    return np.array(embeddings)

def load_embeddings_image(folder_path):
    embeddings = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            embedding = np.load(file_path)
            embedding = embedding.squeeze()  # (1, 3, 224, 224) -> (3, 224, 224)
            embedding = embedding.flatten()  # Flatten to 2048
            embeddings.append(embedding)
    
    return np.array(embeddings)


def generate_embeddings_for_multiple_folders(folders, base_image_path, base_text_path, embed_image_path, embed_text_path,mlp_text, mlp_image):
    all_image_embeddings = []
    all_text_embeddings = []

    for folder in folders:
        image_folder = os.path.join(base_image_path, folder)
        text_folder = os.path.join(base_text_path, folder)

        for file_name in sorted(os.listdir(image_folder)):
            if file_name.endswith((".jpg", ".png")):
                image_path = os.path.join(image_folder, file_name)
                text_file_name = file_name.replace(".jpg", ".txt").replace(".png", ".txt")
                text_path = os.path.join(text_folder, text_file_name)

                if os.path.exists(text_path):
                    # Save text embedding
                    text_save_path = os.path.join(embed_text_path, folder, text_file_name.replace(".txt", ".npy"))
                    process_and_save_text_embedding_with_mlp(text_path, text_save_path,mlp_model=mlp_text)
                if os.path.exists(image_path):
                    # Save image embedding
                    image_save_path = os.path.join(embed_image_path, folder, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))
                    process_and_save_image_embedding_with_mlp(image_path, image_save_path,mlp_model=mlp_image)

        # Load generated embeddings for the folder
        image_embeddings = load_embeddings_image(os.path.join(embed_image_path, folder))
        text_embeddings = load_embeddings(os.path.join(embed_text_path, folder))

        all_image_embeddings.append(image_embeddings)
        all_text_embeddings.append(text_embeddings)

    # Concatenate embeddings from all folders in the group
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    for i, arr in enumerate(all_text_embeddings):
        print(f"[DEBUG-CONCAT] index {i} shape: {arr.shape}, ndim: {arr.ndim}")
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0)

    return all_image_embeddings, all_text_embeddings

def generate_or_load_embeddings_for_group(folders, base_image_path, base_text_path, embed_image_path, embed_text_path, mlp_text, mlp_image):
    all_image_embeddings = []
    all_text_embeddings = []

    # Create MLP-specific folder names
    text_config = format_embed_folder(mlp_text[0].in_features if hasattr(mlp_text[0], 'in_features') else 'unknown', 
                                    mlp_text[-1].out_features, 0.2, True, nn.ReLU)
    image_config = format_embed_folder(mlp_image[0].in_features if hasattr(mlp_image[0], 'in_features') else 'unknown',
                                     mlp_image[-1].out_features, 0.2, True, nn.ReLU)

    for folder in folders:
        # Create MLP-specific paths
        image_embed_folder = os.path.join(embed_image_path, folder, image_config)
        text_embed_folder = os.path.join(embed_text_path, folder, text_config)
        image_folder = os.path.join(base_image_path, folder)
        text_folder = os.path.join(base_text_path, folder)

        # Check if embeddings exist for this specific MLP configuration
        if os.path.exists(image_embed_folder) and any(f.endswith(".npy") for f in os.listdir(image_embed_folder)):
            print(f"Loading cached image embeddings for {folder} with config: {image_config}")
            image_embeddings = load_embeddings_image(image_embed_folder)
        else:
            print(f"Generating new image embeddings for {folder} with config: {image_config}")
            os.makedirs(image_embed_folder, exist_ok=True)
            image_embeddings = generate_and_save_embeddings(image_folder, image_embed_folder, folder, ".jpg", mlp_image)
            image_embeddings = np.array(image_embeddings) # Ensure it's a numpy array

        all_image_embeddings.append(image_embeddings)

        # Check if embeddings exist for this specific MLP configuration
        if os.path.exists(text_embed_folder) and any(f.endswith(".npy") for f in os.listdir(text_embed_folder)):
            print(f"Loading cached text embeddings for {folder} with config: {text_config}")
            text_embeddings = load_embeddings(text_embed_folder)
        else:
            print(f"Generating new text embeddings for {folder} with config: {text_config}")
            os.makedirs(text_embed_folder, exist_ok=True)
            text_embeddings = generate_and_save_embeddings(text_folder, text_embed_folder, folder, ".txt", mlp_text)
            text_embeddings = np.array(text_embeddings) # Ensure it's a numpy array
        
        if text_embeddings.ndim == 1:
            print(f"[ERROR] {folder} 폴더의 임베딩 shape이 1차원: {text_embeddings.shape}")
        else:
            print(f"[OK] {folder} 폴더의 임베딩 shape: {text_embeddings.shape}")
        all_text_embeddings.append(text_embeddings)
        

    # Concatenate embeddings from all folders in the group
    all_image_embeddings = np.concatenate(all_image_embeddings, axis=0) if all_image_embeddings else np.array([])
    all_text_embeddings = np.concatenate(all_text_embeddings, axis=0) if all_text_embeddings else np.array([])

    return all_image_embeddings, all_text_embeddings

def generate_and_save_embeddings(data_folder, embed_path, folder_name, file_extension, mlp_model):
    embeddings = []
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith((".jpg", ".png")) and file_extension == ".jpg":
            image_path = os.path.join(data_folder, file_name)
            save_path = os.path.join(embed_path, file_name.replace(".jpg", ".npy").replace(".png", ".npy"))
            process_and_save_image_embedding_with_mlp(image_path, save_path, mlp_model=mlp_model)
            embedding = np.load(save_path)
            # Ensure shape is (1, D) or (N, D)
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]
            elif embedding.ndim == 3 and embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)
            print(f"[DEBUG] {file_name} image embedding shape: {embedding.shape}")
            embeddings.append(embedding)
        elif file_name.endswith(".txt") and file_extension == ".txt":
            text_path = os.path.join(data_folder, file_name)
            save_path = os.path.join(embed_path, file_name.replace(".txt", ".npy"))
            process_and_save_text_embedding_with_mlp(text_path, save_path, mlp_model=mlp_model)
            embedding = np.load(save_path)
            # Ensure shape is (1, D) or (N, D)
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]
            elif embedding.ndim == 3 and embedding.shape[1] == 1:
                embedding = embedding.squeeze(1)
            print(f"[DEBUG] {file_name} text embedding shape: {embedding.shape}")
            embeddings.append(embedding)
    if embeddings:
        return np.concatenate(embeddings, axis=0)
    else:
        return np.array([])

def generate_all_embeddings(selected_folders, base_image_path, base_text_path, embed_image_path, embed_text_path,mlp_text, mlp_image):
    
    # Group 1: X, V
    image_embeddings_1, text_embeddings_1 = generate_or_load_embeddings_for_group(
        selected_folders[0], base_image_path, base_text_path, embed_image_path, embed_text_path,mlp_text=mlp_text, mlp_image=mlp_image
    )
    X = text_embeddings_1 # Use only text embeddings for X
    V = image_embeddings_1  # Use only image embeddings for V

    # Group 2: Y, W
    image_embeddings_2, text_embeddings_2 = generate_or_load_embeddings_for_group(
        selected_folders[1], base_image_path, base_text_path, embed_image_path, embed_text_path,mlp_text=mlp_text, mlp_image=mlp_image
    )
    Y = text_embeddings_2 # Use only text embeddings for Y
    W = image_embeddings_2  # Use only image embeddings for W

    return X, V, Y, W


# # Randomly select folders
def select_random_folders(bird_list, count):
    return random.sample(bird_list, count)

# Two-sample test logic
def run_tests(K, L, X, Y, V, W, N1, N2, alpha=0.05):
    # Clear GPU cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert numpy arrays to PyTorch tensors
    X_full = torch.tensor(X, dtype=torch.float32)
    Y_full = torch.tensor(Y, dtype=torch.float32)
    V_full = torch.tensor(V, dtype=torch.float32)
    W_full = torch.tensor(W, dtype=torch.float32)
    
    tests = ['mmd-perm', 'xMMD', 'xssMMD(knn)', 'xssMMD(ker)', 'xssMMD(rf)']
    outputs = [[] for _ in range(len(tests))]
    stats = [[] for _ in range(len(tests))]
    total_iterations = K * L
    completed_iterations = 0

    num_perms = 200
    thresh_permutation = partial(get_bootstrap_threshold, num_perms=num_perms)
    thresh_normal = get_normal_threshold
    thresh_spectral = partial(get_spectral_threshold,  alpha=0.05, numNullSamp=200)

    for kk in tqdm(range(K)):
        # Clear GPU cache at the start of each K iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        torch.manual_seed(kk * 19)
        torch.cuda.manual_seed(kk * 19)
        np.random.seed(1102 * (kk + 10))

        # Sample N1 random indices
        idx1 = torch.randperm(X_full.shape[0])[:N1] 
        # Calculate the remaining indices
        all_indices1 = torch.arange(0, X_full.shape[0]) 
        remaining_indices1 = all_indices1[~torch.isin(all_indices1, idx1)]  
        X = X_full[idx1]
        
        V_first_part = V_full[idx1] 
        V_remaining_part = V_full[remaining_indices1]  
        V = torch.cat([V_first_part, V_remaining_part], dim=0) 
            
        for ll in tqdm(range(L)):
            # Sample N1 random indices
            idx2 = torch.randperm(Y_full.shape[0])[:N2] 
            # Calculate the remaining indices
            all_indices2 = torch.arange(0, Y_full.shape[0])  
            remaining_indices2 = all_indices2[~torch.isin(all_indices2, idx2)]  

            Y = Y_full[idx2]
            
            W_first_part = W_full[idx2] 
            W_remaining_part = W_full[remaining_indices2] 
            W = torch.cat([W_first_part, W_remaining_part], dim=0)  
            
            # Kernel bandwidth
            bw = get_median_bw(X=X, Y=Y)
            bw2 = get_median_bw(X=V, Y=W)
            kernel_type = RBFkernel  # RBFkernel, Linearkernel
            kernel_func = None
            if kernel_func is None:  # default is to use the RBF kernel
                if kernel_type == RBFkernel or kernel_type is None:
                    kernel_type = RBFkernel  # just in case it is None
                    kernel_func = partial(RBFkernel1, bw=bw)
                    kernel_func2 = partial(RBFkernel1, bw=bw2)
                elif kernel_type == Linearkernel:
                    kernel_func = partial(Linearkernel)

            unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True)
            biased_mmd2 = partial(TwoSampleMMDSquared, unbiased=False)
            cross_mmd2 = crossMMD2sampleUnpaired
            
            retry = True  

            while retry:
                retry = False  
                # Perform tests
                for i, test in enumerate(tests):
                    try:
                        if test == 'mmd-perm':
                            stat = unbiased_mmd2(X, Y, kernel_func)
                            th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2, alpha=alpha)
                        elif test == 'mmd-perm2':
                            stat = unbiased_mmd2(V, W, kernel_func2)
                            th = thresh_permutation(V, W, kernel_func2, unbiased_mmd2, alpha=alpha)
                        elif test == 'xMMD':
                            stat = cross_mmd2(X, Y, kernel_func)                            
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(knn)':
                            stat = crossSSMMD2sample(X, V, Y, W, kernel_func, "KNN")
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(ker)':
                            stat = crossSSMMD2sample(X, V, Y, W, kernel_func, "KernelRegression")
                            th = thresh_normal(alpha)
                        elif test == 'xssMMD(rf)':
                            stat = crossSSMMD2sample(X, V, Y, W, kernel_func, "RandomForest")
                            th = thresh_normal(alpha)
                        else:
                            raise ValueError(f"Unknown test type: {test}")

                        # Append stat and result
                        stats[i].append(stat)
                        outputs[i].append((stat > th).float().item())
                    except Exception as e:
                        print(f"Error in '{test}' test: {e}")
                        retry = True  # Set retry flag to True to repeat the current iteration
                        break  # Exit the current test and retry

                if retry:
                    print(f"Retrying iteration {kk + 1}, {ll + 1}...")
                    break  # Break the current `ll` and `kk` loop to start the next one
                else:
                    break  # Proceed to the next iteration

            # Increment completed iteration counter
            completed_iterations += 1

            # Clear GPU cache after each L iteration to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Ensure that the total number of iterations reaches K * L
            if completed_iterations >= total_iterations:
                print(f"Completed {total_iterations} iterations.")
                break
        
    # Calculate mean results
    mean_outputs = [np.mean(output) for output in outputs]
    
    # Print results
    for i, test in enumerate(tests):
        print(f"Test: {test}, Mean result: {mean_outputs[i]}")

    return stats, outputs


# Main execution
if __name__ == "__main__":
    # Set your OWN path!
    file_path = '/Users/Downloads/CUB_200_2011/classes.txt'
    bird_list = []
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            bird_name = ' '.join(stripped_line.split(' ')[1:])
            bird_list.append(bird_name)
    print(bird_list)

    bird_insect = [
        '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher', '042.Vermilion_Flycatcher', 
        '102.Western_Wood_Pewee', '154.Red_eyed_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', 
        '159.Black_and_white_Warbler', '161.Blue_winged_Warbler', '167.Hooded_Warbler',  
        '179.Tennessee_Warbler', '181.Worm_eating_Warbler'
    ]
    bird_forest = [
        '038.Great_Crested_Flycatcher', '154.Red_eyed_Vireo', '157.Yellow_throated_Vireo',    
        '161.Blue_winged_Warbler', '179.Tennessee_Warbler', '042.Vermilion_Flycatcher',     
        '139.Scarlet_Tanager', '094.White_breasted_Nuthatch', '133.White_throated_Sparrow'    
    ]
    
    bird_fish = [
        '079.Belted_Kingfisher','080.Green_Kingfisher', '081.Pied_Kingfisher',      
        '082.Ringed_Kingfisher', '052.Pied_billed_Grebe', '143.Caspian_Tern',         
         '129.Song_Sparrow'
    ]
    bird_wetland = [
        '079.Belted_Kingfisher', '080.Green_Kingfisher', '081.Pied_Kingfisher',      
        '082.Ringed_Kingfisher', '052.Pied_billed_Grebe', '143.Caspian_Tern', '087.Mallard' 
    ]
    
    bird_seed = [
        '017.Cardinal', '076.Dark_eyed_Junco', '118.House_Sparrow', '120.Fox_Sparrow', '129.Song_Sparrow', 
        '133.White_throated_Sparrow', '055.Evening_Grosbeak'
    ] 
    bird_scrub = [
        '017.Cardinal', '076.Dark_eyed_Junco', '118.House_Sparrow', '120.Fox_Sparrow', 
        '129.Song_Sparrow', '133.White_throated_Sparrow', 
        '094.White_breasted_Nuthatch'
    ] 
    
    bird_sparrow = [
        '113.Baird_Sparrow', '114.Black_throated_Sparrow',	'115.Brewer_Sparrow',
        '116.Chipping_Sparrow','117.Clay_colored_Sparrow','118.House_Sparrow','119.Field_Sparrow',
        '120.Fox_Sparrow','121.Grasshopper_Sparrow','122.Harris_Sparrow','123.Henslow_Sparrow',
        '129.Song_Sparrow','130.Tree_Sparrow','131.Vesper_Sparrow','132.White_crowned_Sparrow'
    ]
    bird_gound = [
        '121.Grasshopper_Sparrow','122.Harris_Sparrow','130.Tree_Sparrow','116.Chipping_Sparrow',
        '129.Song_Sparrow','131.Vesper_Sparrow','132.White_crowned_Sparrow','076.Dark_eyed_Junco',
        '021.Eastern_Towhee','148.Green_tailed_Towhee','149.Brown_Thrasher','150.Sage_Thrasher',
        '193.Bewick_Wren','196.House_Wren','199.Winter_Wren','028.Brown_Creeper','099.Ovenbird',
        '181.Worm_eating_Warbler'
    ]
    
    bird_cuckoo = [
        '031.Black_billed_Cuckoo', '033.Yellow_billed_Cuckoo', '004.Groove_billed_Ani', 
        '154.Red_eyed_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', 
        '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler', '182.Yellow_Warbler'
    ]
    bird_foliage = [
        '031.Black_billed_Cuckoo', '033.Yellow_billed_Cuckoo', '004.Groove_billed_Ani', 
        '154.Red_eyed_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', 
        '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler', '182.Yellow_Warbler',
        '021.Eastern_Towhee', '188.Pileated_Woodpecker', '133.White_throated_Sparrow'    
    ]

    bird_warbler = [
        '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler',
        '161.Blue_winged_Warbler', '162.Canada_Warbler', '164.Cerulean_Warbler', 
        '167.Hooded_Warbler', '168.Kentucky_Warbler', '169.Magnolia_Warbler', '170.Mourning_Warbler',
        '176.Prairie_Warbler', '179.Tennessee_Warbler', '180.Wilson_Warbler', '182.Yellow_Warbler'
    ]
    bird_canopy = [
        '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler',
        '161.Blue_winged_Warbler', '162.Canada_Warbler', '164.Cerulean_Warbler', '167.Hooded_Warbler',
        '169.Magnolia_Warbler', '170.Mourning_Warbler', '182.Yellow_Warbler', 
        '154.Red_eyed_Vireo', '038.Great_Crested_Flycatcher' 
    ]

    # Select bird folders
    # selected_folders_updated = [bird_insect, bird_forest]
    # selected_folders_updated = [bird_fish, bird_wetland] 
    # selected_folders_updated = [bird_seed, bird_scrub] 
    # selected_folders_updated = [bird_sparrow, bird_goudn]
    selected_folders_updated = [bird_cuckoo, bird_foliage]
    # selected_folders_updated = [bird_warbler, bird_canopy]
    print(f"Selected folders: {selected_folders_updated}") 

    
    # Set your OWN local path
    base_image_path = "/Users/Downloads/CUB_200_2011/images"
    base_text_path = "/Users/Downloads/CUB_200_2011/text_c10/"
    
    # Define MLP parameters
    hidden_dims_text = [[128]] 
    output_dims_text = [4] 
    dropouts_text = [0.2]
    batchnorms_text = [True]
    activations_text = [nn.ReLU]

    hidden_dims_image = [[1024, 256]] 
    output_dims_image = [32]
    dropouts_image = [0.2]
    batchnorms_image = [True]
    activations_image = [nn.ReLU]

    for h_text, o_text, d_text, b_text, a_text, h_img, o_img, d_img, b_img, a_img in itertools.product(
        hidden_dims_text, output_dims_text, dropouts_text, batchnorms_text, activations_text,
        hidden_dims_image, output_dims_image, dropouts_image, batchnorms_image, activations_image
    ):
        # Clear GPU cache at the start of each parameter combination
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Set your OWN path
        param_folder = f"text_{format_embed_folder(h_text, o_text, d_text, b_text, a_text)}__" \
                       f"image_{format_embed_folder(h_img, o_img, d_img, b_img, a_img)}"
        base_embed_path = "/Users/Downloads/CUB_200_2011/cub_bert_embedding"
        
        embed_image_path = os.path.join(base_embed_path, param_folder, "image")
        embed_text_path = os.path.join(base_embed_path, param_folder, "text")

        mlp_text = make_mlp(input_dim=768, hidden_dims=h_text, output_dim=o_text,
                            dropout=d_text, batchnorm=b_text, activation=a_text)
        mlp_image = make_mlp(input_dim=2048, hidden_dims=h_img, output_dim=o_img,
                             dropout=d_img, batchnorm=b_img, activation=a_img)

        # Generate embeddings
        X, V, Y, W = generate_all_embeddings(
            selected_folders_updated, 
            base_image_path, 
            base_text_path, 
            embed_image_path, 
            embed_text_path, 
            mlp_text=mlp_text, 
            mlp_image=mlp_image
        )

        # Print shapes
        print(f"X shape: {X.shape}, V shape: {V.shape}")
        print(f"Y shape: {Y.shape}, W shape: {W.shape}")

        # Run two-sample test
        stats, outputs = run_tests(K=50, L=20, X=X, Y=Y, V=V, W=W, N1=150, N2=150)

        # Save results
        os.makedirs("results", exist_ok=True)
        # Extract first string from each folder group
        first_folder_first_bird = selected_folders_updated[0][0]  # First bird from first folder group
        second_folder_first_bird = selected_folders_updated[1][0]  # First bird from second folder group
        
        # Add timestamp to experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"results_{first_folder_first_bird}_{second_folder_first_bird}_{h_text}_{o_text}_{d_text}_{b_text}_{a_text.__name__}_{h_img}_{o_img}_{d_img}_{b_img}_{a_img.__name__}_{timestamp}"
        np.save(f"results/stats_{exp_name}.npy", to_cpu_numpy(stats), allow_pickle=True)
        np.save(f"results/outputs_{exp_name}.npy", to_cpu_numpy(outputs), allow_pickle=True)