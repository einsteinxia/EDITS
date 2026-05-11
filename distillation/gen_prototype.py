from diffusers import StableDiffusionGenLatentsPipeline
from diffusers import AutoencoderKL
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import normalize
import torch
import torchvision  
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import argparse
import json
import numpy as np
import math
import os
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from classes import IMAGENET2012_CLASSES
from tiny_imagenet_classes import tiny_imagenet_CLASSES
from dataset_utils import *
import ipdb
from collections import Counter
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import CLIPProcessor, CLIPModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.set_proxy('https://pypi.tuna.tsinghua.edu.cn/simple') 
# nltk.download('punkt_tab')
# nltk.download('stopwords')
from collections import defaultdict
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
import httpx
import json
from openai import OpenAI
import time
from typing import List, Dict
from classes import IMAGENET2012_CLASSES

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=10, type=int, 
                        help='batch size')
    parser.add_argument('--threshold', default=0.7, type=float, 
                        help='threshold')
    parser.add_argument('--tpk', default=20, type=int, 
                        help='topcommon words')
    parser.add_argument('--data_dir', default='data/imagenet', type=str, 
                        help='root dir')
    parser.add_argument('--dataset', default='imagenet', type=str, 
                        help='data prepare to distillate:imagenet/tiny-imagenet')
    parser.add_argument('--diffusion_checkpoints_path', default="stablediffusion/checkpoints/stable-diffusion-v1-5", type=str, 
                        help='path to stable diffusion model from pretrained')
    parser.add_argument('--vae_checkpoints_path', default="pretrain_model/sd-vae-ft-mse", type=str, 
                        help='path to VAE model from pretrained')
    parser.add_argument('--clip_checkpoints_path', default="pretrain_model/CLIP", type=str, 
                        help='path to CLIP model from pretrained')
    parser.add_argument('--ipc', default=1, type=int, 
                        help='image per class')
    parser.add_argument('--metajson_file', type=str, 
                        help='metajson_file')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='contamination')
    parser.add_argument('--km_expand', default=10, type=int, 
                        help='expand ration for minibatch k-means model')
    parser.add_argument('--label_file_path', default='data/imagenet_classes.txt', type=str, 
                        help='root dir')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='number of workers')
    parser.add_argument('--save_prototype_path', default='prototypes', type=str, 
                        help='where to save the generated prototype json files')
    parser.add_argument('--seed', default=0, type=int, 
                        help='seed')              
    parser.add_argument('--size', default=256, type=int, 
                        help='init resolution (resize)')
    parser.add_argument('--spec', type=str, 
                        help='')
    parser.add_argument('--method', default='ours', type=str, 
                        help='method')
    parser.add_argument('--feature_normlize', default=1, type=int, 
                        help='feature_normlize')
    parser.add_argument('--tau', default=0.01, type=float, 
                        help='temperature tau for softmax')
    parser.add_argument('--api_key',  type=str, 
                    help='your api key for LLM')
    parser.add_argument('--max_tokens',  type=int, 
                    help='')
    parser.add_argument('--k_text', default=1, type=int, 
                    help='')
    parser.add_argument('--k_image', default=1, type=int, 
                    help='')

    args = parser.parse_args()
    return args



def build_prompt(texts: List[str], ground_truth, max_tokens: int) -> str:
    """Build the prompt for prototype text generation"""
    
    text_examples = "\n\n".join([f"[Text {i+1}]\n{text}" for i, text in enumerate(texts)])
    
    prompt = f"""Please analyze the following {len(texts)} texts and generate a high-quality representative prototype text.
    ## Input Texts:
    {text_examples}

    ## Output Requirements:
    1. Extract semantic content directly related to label {ground_truth} (such as classification tags, keywords, topic identifiers, etc.) in each text.
    2. Abstract the semantics related to label {ground_truth} into universal expressions, avoid relying on the sentence structure or vocabulary of specific texts, and ensure that the prototype text covers the commonality of label {ground_truth} in all texts.
    3. Merge unique information and expressions from each text.
    4. The generated text should represent the core content of all input texts.
    5. Output must be limited to {max_tokens} tokens.
    6. Fluent language, accurate information, clear structure.
    7. If there is a specific animal species, it must be reflected in the generated text and cannot be mentioned in general terms (for example, "Dingo" cannot be just called "dog").
    OUTPUT ONLY THE PROTOTYPE TEXT. """
    return prompt



def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj
    
def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def initialize_km_models(label_list, args):
    km_models = {}
    for prompt in label_list:
        model_name = f"KMeans_{prompt}"
        model = KMeans(n_clusters=args.ipc, random_state=args.seed, n_init=10)
        km_models[model_name] = model
    return km_models

def prototype_kmeans_ours(vae, encode_model, processor, data_loader, label_list, km_models, args):
    "Code is coming soon..."
        latents = {label: [] for label in label_list} 
    embeds = {label: [] for label in label_list}
    prompt_to_paths = {label: [] for label in label_list} 
    fuse_feature = {label: [] for label in label_list}
    print('Extracting image and text features for all training data...')
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), position=0):
    
        images = batch['images'].cuda(non_blocking=True)
        batch_paths = batch['image_paths']
        labels = batch['labels'].cuda(non_blocking=True)
        texts = batch['texts']
        # label_str = batch['label_strs']
        # indices = batch['indexs']
        
        prompts = []
        
        for idx, label in enumerate(labels):
            
            prompt = label_list[label.item()]
            prompts.append(prompt)
            prompt_to_paths[prompt].append(batch_paths[idx])

        
        # # vae    
        # init_latents = vae.encode(F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)).latent_dist.mean * vae.config.scaling_factor 
        # # init_latents, init_embeds = pipe(prompt=text_prompt, negative_prompt=[negative_prompt for i in range(len(text_prompt))], image=images, strength=0.7, guidance_scale=8)
        # # print('init_latents.shape:', init_latents.shape)
        # # print('init_embeds.shape:', init_embeds.shape)

        # for latent, prompt in zip(init_latents, prompts):
        #     latents[prompt].append(latent.view(-1).cpu().detach().numpy())
        
        # clip
        init_latents = CLIP_Image_Embeddings(args, images, encode_model, processor)
        init_embed = CLIP_Text_Embeddings(args, texts, encode_model, processor)

        if args.feature_normlize: 
            init_latents / np.linalg.norm(init_latents, axis=1, keepdims=True)
            init_embed / np.linalg.norm(init_embed, axis=1, keepdims=True)
            
        for latent, embed, prompt in zip(init_latents, init_embed, prompts):
            latents[prompt].append(latent)
            embeds[prompt].append(embed)
            
               
    for prompt in label_list:
        prompt_latents= latents[prompt]
        prompt_embeds = embeds[prompt]    
        
        # concat_features
        # fuse_feature[prompt] = np.concatenate([prompt_latents, prompt_embeds], axis=1)
        
        # weighted_fuse_features  
        # prompt_latents = normalize(np.array(prompt_latents), norm='l2')
        # prompt_latents = normalize(np.array(prompt_embeds), norm='l2')  
        similarity = np.matmul(np.array(prompt_latents), np.array(prompt_embeds).T)
        # np.fill_diagonal(similarity, 1.0)
        similarity = softmax(similarity / args.tau, axis=1)
        weight_embedding = np.matmul(similarity, prompt_embeds)
        fuse_feature[prompt] = np.concatenate([prompt_latents, weight_embedding], axis=1)
    
    # # only image feature
    # fuse_feature = latents
    
    
    
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_all_image_latent.npy')
    # print('latents:', latents)
    # print('type(latents):', type(latents))
    np.save(os.path.join(json_file), fuse_feature, allow_pickle=True)
    # with open(json_file, 'w') as f:
    #     # json.dump(convert_to_serializable(latents), f)
    #     json.dump(latents, f, default=lambda x: x.tolist())
        # json.dump({k: [arr.tolist() for arr in v] for k, v in latents.items()}, f)
    print(f"all image latent json file saved ")
    del init_latents, prompts
    final_latents = {label: [] for label in label_list}
    for prompt in label_list:
        if len(fuse_feature[prompt]) >= args.ipc:
            if args.contamination == 0:
                inliers = [True for i in range(len(fuse_feature[prompt]))]
            else:
                clf = LocalOutlierFactor(n_neighbors=10, contamination=args.contamination)
                X_train = np.vstack(fuse_feature[prompt])
                y_pred = clf.fit_predict(X_train)
                inliers = y_pred == 1
            num_false = np.sum(inliers == False)
            # print(f'-------------{inliers}--------------{len(latents[prompt])}--------------{num_false}')
            fuse_feature[prompt] = np.array(fuse_feature[prompt])[inliers].tolist()
            final_latents[prompt] = fuse_feature[prompt].copy()
            # print(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx{len(latents[prompt])}')
            prompt_to_paths[prompt] = np.array(prompt_to_paths[prompt])[inliers].tolist()
            km_models[f"KMeans_{prompt}"].fit(np.vstack(fuse_feature.pop(prompt,None)))

    return km_models, prompt_to_paths, final_latents

def find_closest_samples(cluster_centers, final_latents, multi_samples_per_cluster=3):
    distance_matrix = euclidean_distances(cluster_centers, final_latents)
    n_clusters = cluster_centers.shape[0]
    
    sorted_indices = np.argsort(distance_matrix, axis=1)
    
    single_indices = []
    used_mask = np.zeros(final_latents.shape[0], dtype=bool)
    
    for i in range(n_clusters):
        for idx in sorted_indices[i]:
            if not used_mask[idx]:
                single_indices.append(idx)
                used_mask[idx] = True
                break
    
    single_indices = np.array(single_indices)
    
    multi_indices = np.zeros((n_clusters, multi_samples_per_cluster), dtype=int)
    for i in range(n_clusters):
        cluster_sorted = sorted_indices[i]
        selected = cluster_sorted[:multi_samples_per_cluster]
        multi_indices[i] = selected
    
    single_samples = final_latents[single_indices]
    multi_samples = final_latents[multi_indices]
    
    return single_indices, single_samples, multi_indices, multi_samples


def find_cluster_samples(centers, labels, samples, k_text=None, k_image=None):
    all_labels = labels.tolist()
    
    results = {
        'nearest_indices': {},          
        'nearest_samples': {},          
        'nearest_k_indices_text': {},       
        'nearest_k_samples_text': {},    
        'nearest_k_indices_image': {},      
        'nearest_k_samples_image': {}, 
        'all_cluster_indices': {},      
        'all_cluster_samples': {}  
    }
    
    n_clusters = len(centers)
    
    cluster_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        cluster_indices[label].append(idx)
    
    for cluster_id in range(n_clusters):
        current_cluster_indices = cluster_indices.get(cluster_id, [])
        
        if not current_cluster_indices:
            continue
        
        cluster_samples = samples[current_cluster_indices]
        center = centers[cluster_id]
        distances = np.linalg.norm(cluster_samples - center, axis=1)
        
        sorted_indices = np.argsort(distances)
        
        nearest_idx = sorted_indices[0]
        original_nearest_idx = current_cluster_indices[nearest_idx]
        results['nearest_indices'][cluster_id] = original_nearest_idx
        results['nearest_samples'][cluster_id] = cluster_samples[nearest_idx]
        
        if k_text is not None:
            k_nearest_indices = sorted_indices[:min(k_text, len(sorted_indices))]
            original_k_indices = [current_cluster_indices[i] for i in k_nearest_indices]
            results['nearest_k_indices_text'][cluster_id] = original_k_indices
            results['nearest_k_samples_text'][cluster_id] = cluster_samples[k_nearest_indices]
        else:
            results['nearest_k_indices_text'][cluster_id] = None
            results['nearest_k_samples_text'][cluster_id] = None
            
        if k_image is not None:
            k_nearest_indices = sorted_indices[:min(k_image, len(sorted_indices))]
            original_k_indices = [current_cluster_indices[i] for i in k_nearest_indices]
            results['nearest_k_indices_image'][cluster_id] = original_k_indices
            results['nearest_k_samples_image'][cluster_id] = cluster_samples[k_nearest_indices]
        else:
            results['nearest_k_indices_image'][cluster_id] = None
            results['nearest_k_samples_image'][cluster_id] = None
        
        results['all_cluster_indices'][cluster_id] = current_cluster_indices
        results['all_cluster_samples'][cluster_id] = cluster_samples
    
    return results

def gen_prototype_ours(label_list, km_models, prompt_to_paths, final_latents, args):
    stop_words = set(stopwords.words('english'))
    # print('prompt_to_paths:', prompt_to_paths)
    text_dic = {}
    with open(args.metajson_file, 'r') as f:
        for line in f:
            json_data = json.loads(line.strip())
            file_name = json_data['file_name']
            text = json_data['text']
            text_dic[file_name] = text
    # print('data_dict:', data_dict)
    image_pro = {}
    image_multi_pro = {}
    img_pro_path = {}
    text_pro = {label: [] for label in label_list}
    text_multi_pro = {}
    all_labels = {label: [] for label in label_list}
    all_path = {label: [] for label in label_list}
    samples_per_cluster = {}
    image_multi_pro_path = {label: [] for label in label_list}
    # api 
    client = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    print("generateing prototype......")
    for prompt in tqdm(label_list):
        model_name = f"KMeans_{prompt}"
        model = km_models.pop(model_name,None)
        labels = model.labels_
        all_labels[prompt].append(labels)
        all_path[prompt].append(prompt_to_paths[prompt])
        cluster_centers = model.cluster_centers_
        num_clusters = cluster_centers.shape[0]
         
        results = find_cluster_samples(cluster_centers, labels, np.array(final_latents[prompt]), k_text=args.k_text, k_image=args.k_image)
        
        closest_indices = list(results['nearest_indices'].values())
        n_closest_indices_text = list(results['nearest_k_indices_text'].values())
        n_cluster_centers_text = [value.tolist() for value in results['nearest_k_samples_text'].values()]
        n_closest_indices_image = list(results['nearest_k_indices_image'].values())
        n_cluster_centers_image = [value.tolist() for value in results['nearest_k_samples_image'].values()]
        all_closest_indices = list(results['all_cluster_indices'].values())
        
        # image prototype
        image_pro[prompt] = [cluster_centers[i].tolist() for i in range(num_clusters)] 
        image_multi_pro[prompt] = [n_cluster_centers_image[i] for i in range(num_clusters)]
        img_pro_path[prompt] = [prompt_to_paths[prompt][i] for i in closest_indices]
        image_multi_pro_path[prompt] = [[prompt_to_paths[prompt][idx] for idx in cluster_indices] for cluster_indices in n_closest_indices_image]
        
        # text prototype 
        
        ## multi-one mapping
        text_multi_pro_path = [[prompt_to_paths[prompt][idx] for idx in cluster_indices] for cluster_indices in n_closest_indices_text]
        text_multi_pro[prompt] = [[text_dic.get(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", "unknown text") for path in cluster_paths] for cluster_paths in text_multi_pro_path]
        
        """Process multiple text groups in batch --> llm api"""
        for _, texts in enumerate(text_multi_pro[prompt], 1):
            # print('IMAGENET2012_CLASSES[prompt]:', IMAGENET2012_CLASSES[prompt])
            llm_prompt = build_prompt(texts, IMAGENET2012_CLASSES[prompt], args.max_tokens)
            result = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a professional text analysis expert skilled at extracting core information from multiple texts and generating high-quality representative prototype text. Strictly adhere to token limits."},
                    {"role": "user", "content": llm_prompt},
                ],
                stream=False
            )

            text_pro[prompt].append(result.choices[0].message.content)       

        # # one-one mapping
        # text_pro[prompt] = [text_dic[prompt_to_paths[prompt][i].split('train/')[1]] for i in closest_indices]
   
    # json_file_cluster_all_latent = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_all_latent.json')
    # with open(json_file_cluster_all_latent, 'w') as f:
    #     json.dump(final_latents, f)
    
    json_file_cluster_image_pro = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_image_pro.json')
    with open(json_file_cluster_image_pro, 'w') as f:
        json.dump(image_pro, f) 
        
    # json_file_cluster_image_multi_pro = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_image_multi_pro.json')
    # with open(json_file_cluster_image_multi_pro, 'w') as f:
    #     json.dump(image_multi_pro, f)
    
    json_file_cluster_image_pro_path = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_image_pro_path.json')
    with open(json_file_cluster_image_pro_path, 'w') as f:
        json.dump(img_pro_path, f) 
        
    json_file_cluster_image_all_path = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_image_all_path.json')
    with open(json_file_cluster_image_all_path, 'w') as f:
        json.dump(image_multi_pro_path, f) 
        
    json_file_cluster_text_pro = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_text_pro.json')
    with open(json_file_cluster_text_pro, 'w') as f:
        json.dump(text_pro, f) 
    
    
    json_file_all_path = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_all_path.json')
    with open(json_file_all_path, 'w') as f:
        json.dump(all_path, f)
        
    json_file_cluster_text_multi_pro = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_text_multi_pro.json')
    with open(json_file_cluster_text_multi_pro, 'w') as f:
        json.dump(text_multi_pro, f)
        
    # json_file_cluster_all_label = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_all_label.json')
    # with open(json_file_cluster_all_label, 'w') as f:
    #     json.dump(all_labels, f) 
        
    json_file_labels = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-cluster_all_labels.npy')
    np.save(json_file_labels, all_labels, allow_pickle=True) 

    print(f"all image-text-prototype json file saved ")
              
    return image_pro, text_pro

def prototype_kmeans(pipe, data_loader, label_list, km_models, args):
    
    latents = {label: [] for label in label_list} 
    embeds = {label: [] for label in label_list}
    prompt_to_paths = {label: [] for label in label_list} 
    fuse_feature = {label: [] for label in label_list}
    for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), position=0):
    
        images = batch['images'].cuda(non_blocking=True)
        batch_paths = batch['image_paths']
        labels = batch['labels'].cuda(non_blocking=True)
        text_prompt = batch['texts']
        prompts = []
        for idx, label in enumerate(labels):
            
            prompt = label_list[label.item()]
            prompts.append(prompt)   
            prompt_to_paths[prompt].append(batch_paths[idx])
        
        negative_prompt = 'cartoon, anime, painting'
        # print(images.shape)

        init_latents, init_embeds = pipe(prompt=text_prompt, negative_prompt=[negative_prompt for i in range(len(text_prompt))], image=images, strength=0.7, guidance_scale=8)

        for latent, embed, prompt in zip(init_latents, init_embeds, prompts):
            latents[prompt].append(latent.view(-1).cpu().numpy())
            embeds[prompt].append(embed.cpu().numpy())
            
    # # concat feature        
    # for prompt in label_list:
    #     prompt_latents= latents[prompt]
    #     prompt_embeds = embeds[prompt] 
    #     fuse_feature[prompt] = np.concatenate([prompt_latents, prompt_embeds], axis=1)
        
    fuse_feature = latents   
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-all_image_latent.npy')
    np.save(json_file, latents, allow_pickle=True)
    print(f"all image latent json file saved ")
    del init_latents,prompts
    final_latents = {label: [] for label in label_list}
    for prompt in label_list:
        if len(fuse_feature[prompt]) >= args.ipc:
            if args.contamination == 0:
                inliers = [True for i in range(len(fuse_feature[prompt]))]
            else:
                clf = LocalOutlierFactor(n_neighbors=10, contamination=args.contamination)
                X_train = np.vstack(fuse_feature[prompt])
                y_pred = clf.fit_predict(X_train)
                inliers = y_pred == 1
            num_false = np.sum(inliers == False)
            # print(f'-------------{inliers}--------------{len(latents[prompt])}--------------{num_false}')
            fuse_feature[prompt] = np.array(fuse_feature[prompt])[inliers].tolist()
            final_latents[prompt] = fuse_feature[prompt].copy()
            # print(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx{len(latents[prompt])}')
            prompt_to_paths[prompt] = np.array(prompt_to_paths[prompt])[inliers].tolist()
            km_models[f"KMeans_{prompt}"].fit(np.vstack(fuse_feature.pop(prompt,None)))
    
    return km_models, prompt_to_paths, final_latents


def find_max_word_sentence(word_weight_pairs, sen):
    max_score = 0
    max_sentence = ""
    
    for sentence in sen:
        score = sum(weight for w, weight in word_weight_pairs if w in sentence)
        
        if score > max_score:
            max_score = score
            max_sentence = sentence
            
    return max_sentence, max_score


def gen_prototype_vlcp(label_list, km_models,prompt_to_paths, final_latents, args):
    # print('prompt_to_paths:', prompt_to_paths)
    data_dict = {}
    stop_words = set(stopwords.words('english'))

    with open(args.metajson_file, 'r') as f:
        for line in f:
            json_data = json.loads(line.strip())
            file_name = json_data['file_name']
            text = json_data['text']
            data_dict[file_name] = text
    # print('data_dict:', data_dict)
    prototype = {}
    near_prototype = {}
    adict = {label: [] for label in label_list}
    all_labels = {label: [] for label in label_list}
    all_path = {label: [] for label in label_list}
    samples_per_cluster = {} 
    images_per_cluster = {}
    img_pro_path = {}
    path_per_cluster = {}
    text_multi_pro = {}
    
    print("generateing prototype......")
    for prompt in tqdm(label_list):
        model_name = f"KMeans_{prompt}"
        model = km_models.pop(model_name,None)
        labels = model.labels_
        all_labels[prompt].append(labels)
        all_path[prompt].append(prompt_to_paths[prompt])
        cluster_centers = model.cluster_centers_
        
        N = int(math.sqrt(cluster_centers.shape[1] / 4))
        num_clusters = cluster_centers.shape[0]
        reshaped_centers = []
        for i in range(num_clusters):
            reshaped_center = cluster_centers[i].reshape(4, N, N)
            reshaped_centers.append(reshaped_center.tolist())
        prototype[prompt] = reshaped_centers
        # prototype[prompt] = cluster_centers
        
        
        
        # just one-one mapping
        closest_indices, clost_samples, n_closest_indices, n_cluster_centers = find_closest_samples(cluster_centers, np.array(final_latents[prompt]), 5)
        adict[prompt] = [data_dict[prompt_to_paths[prompt][i].split('train/')[1]] for i in closest_indices.tolist()]
        img_pro_path[prompt] = [prompt_to_paths[prompt][i] for i in closest_indices.tolist()]
        near_prototype[prompt] = [clost_samples[i].reshape(4, N, N).tolist() for i in range(num_clusters)]
        
        # # ## multi-one mapping
        # text_multi_pro_path = [[prompt_to_paths[prompt][idx] for idx in cluster_indices] for cluster_indices in n_closest_indices]
        # text_multi_pro[prompt] = [[data_dict.get(f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}", "unknown text") for path in cluster_paths] for cluster_paths in text_multi_pro_path]
        
        # """Process multiple text groups in batch --> llm api"""
        # for _, texts in enumerate(text_multi_pro[prompt], 1):
        #     llm_prompt = build_prompt(texts, args.max_tokens)
        #     result = client.chat.completions.create(
        #         model="deepseek-chat",
        #         messages=[
        #             {"role": "system", "content": "You are a professional text analysis expert skilled at extracting core information from multiple texts and generating high-quality representative prototype text. Strictly adhere to token limits."},
        #             {"role": "user", "content": llm_prompt},
        #         ],
        #         stream=False
        #     )

        #     adict[prompt].append(result.choices[0].message.content)
        
        
        # vlcp
        samples_per_cluster[prompt] = {i: [] for i in range(num_clusters)}
        images_per_cluster[prompt] = {i: [] for i in range(num_clusters)}
        path_per_cluster[prompt] = {i: [] for i in range(num_clusters)}
        class_path = prompt_to_paths.pop(prompt,None)
        for idx, label in enumerate(labels):
            sample = class_path[idx]
            new_paths = sample.split('train/', 1)[1]
            # if 'woof' in args.label_file_path: 
            #     new_paths = sample.split('/')[-1]  
            text_desc = data_dict.pop(new_paths, None)
            samples_per_cluster[prompt][label].append(text_desc)
            path_per_cluster[prompt][label].append(new_paths)
        # print('samples_per_cluster:', samples_per_cluster ) 
   
        text_list = []
        word_in_sentence_count_cluster = defaultdict(int)
        for i in range(num_clusters):
            descriptions = samples_per_cluster[prompt][i]
            for sentence in descriptions:
                # print('sentence:', sentence)
                tokens = word_tokenize(sentence)
                words = [word.lower() for word in tokens if word.isalpha()]
                tmp_words = set(words)
                for word in tmp_words:
                    word_in_sentence_count_cluster[word] += 1
        threshold = args.threshold * len(labels)
        cluster_common_text = [word for word, count in word_in_sentence_count_cluster.items() if count >= threshold and word not in stop_words]
        print(f'---{threshold/len(labels)}-------------{len(labels)}--------------------------{cluster_common_text}')
        text_list = []
        for i in range(num_clusters):
            descriptions = samples_per_cluster[prompt][i]
            all_words = []
            for sentence in descriptions:
                tokens = word_tokenize(sentence)
                words = [word.lower() for word in tokens if word.isalpha()]
                all_words.extend(words)
            if args.dataset in ['cifar10','cifar100']:
                filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words and word not in prompt and word not in cluster_common_text]
            elif args.dataset == 'tiny_imagenet':
                filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words and word not in tiny_imagenet_CLASSES[prompt] and word not in cluster_common_text]
            
            else:
                filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words and word not in IMAGENET2012_CLASSES[prompt] and word not in cluster_common_text]
            word_freq = Counter(filtered_words)
            high_freq_words = [(word,freq,len(descriptions)) for word, freq in word_freq.most_common(20)]
            high_freq_words_tmp = [(word,freq) for word, freq in word_freq.most_common(args.tpk)]
            max_sentence, _ = find_max_word_sentence(high_freq_words_tmp, descriptions)
            text_list.append(max_sentence)
            # print("\nfiltered_words:\n", high_freq_words)
            # print("\nGenerated Text:\n", max_sentence)
        adict[prompt]=text_list
        
    
    json_file_labels = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-all_labels.npy')
    np.save(json_file_labels, all_labels, allow_pickle=True)    
        
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-text-vclp.json')
    with open(json_file, 'w') as f:
        json.dump(adict, f)
    print(f"Text json file saved ")
    
    json_file_near_rototype = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-image-near-vclp.json')
    with open(json_file_near_rototype, 'w') as f:
        json.dump(near_prototype, f)
    
    json_file_image_pro_path = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-image_pro_path_vlcp.json')
    with open(json_file_image_pro_path, 'w') as f:
        json.dump(img_pro_path, f)
        
    json_file_all_path = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-all_path.json')
    with open(json_file_all_path, 'w') as f:
        json.dump(all_path, f)
     
    json_file_samples_per_cluster = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-samples_per_cluster.json')
    with open(json_file_samples_per_cluster, 'w') as f:
        json.dump(samples_per_cluster, f)
        
    json_file_path_per_cluster = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-path_per_cluster.json')
    with open(json_file_path_per_cluster, 'w') as f:
        json.dump(path_per_cluster, f)
    # print('len_samples_per_cluster:', len(samples_per_cluster))
    # print(f"all cluster-label-text json file saved ")
              
    return prototype

def save_prototype(prototype, args):
    os.makedirs(args.save_prototype_path, exist_ok=True)
    json_file = os.path.join(args.save_prototype_path, f'{args.dataset}-ipc{args.ipc}-{args.threshold}-{args.tpk}-kmexpand{args.km_expand}-image-vclp.json')
    with open(json_file, 'w') as f:
        json.dump(prototype, f)
    print(f"prototype json file saved at: {args.save_prototype_path}")


def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(prompt.split(" "))
    count_negative_prompt = len(negative_prompt.split(" "))

    # create the tensor based on which prompt is longer
    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                          max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                       max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

def CLIP_Text_Embeddings(args, texts, model, processor):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(args.device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    text_features = text_features.cpu().numpy()
    
    return text_features

def CLIP_Image_Embeddings(args, images, model, processor, is_path=False):
    def apply_clip_normalization(tensor_images):
        if tensor_images.max() > 1.0:
            tensor_images = tensor_images / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        
        mean = mean.view(1, 3, 1, 1).to(tensor_images.device)
        std = std.view(1, 3, 1, 1).to(tensor_images.device)
        
        return (tensor_images - mean) / std
    if is_path:
        processed_images = []
        for img_path in images:
            image = Image.open(img_path)
            processed = processor(images=image, return_tensors="pt")
            processed_images.append(processed['pixel_values'])
    
        images = torch.cat(processed_images, dim=0).cuda(args.device)
    else:
        images = apply_clip_normalization(images).cuda(args.device)
    
    with torch.no_grad():
        image_features = model.vision_model(pixel_values=images).last_hidden_state
        image_features = image_features[:, 0, :]  
        image_features = model.visual_projection(image_features)
        image_features = image_features.cpu().numpy()
    
    return image_features
    
def main():
    args = parse_args()
    set_random_seed(args.seed) 
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1.obtain label-prompt list
    label_list = gen_label_list(args)
    
    # 2.obtain training data
    trainloader = load_dataset_ours(args)
    
    print('args.method:', args.method)
    
    if args.method == 'ours': 
        # 3.define the VAE and CLIP model 
        vae_model = AutoencoderKL.from_pretrained(args.vae_checkpoints_path).to(args.device)
        
        clip_model = CLIPModel.from_pretrained(args.clip_checkpoints_path).to(args.device)
        processor = CLIPProcessor.from_pretrained(args.clip_checkpoints_path)
        clip_model.eval()
        
        # 4.initialize & run partial k-means model each class
        km_models = initialize_km_models(label_list, args)
        fitted_km, prompt_to_paths, final_latents = prototype_kmeans_ours(vae=vae_model, encode_model=clip_model, processor=processor, data_loader=trainloader, label_list=label_list, km_models=km_models, args=args)
        
        # 5.generate prototypes and save them as json file
        gen_prototype_ours(label_list, fitted_km, prompt_to_paths, final_latents, args)
    else:
        # 3.define the diffusers pipeline
        pipe = StableDiffusionGenLatentsPipeline.from_pretrained(args.diffusion_checkpoints_path, torch_dtype=torch.float16)
        pipe = pipe.to(args.device)

        # 4.initialize & run partial k-means model each class
        km_models = initialize_km_models(label_list, args)
        
        fitted_km, prompt_to_paths, final_latents = prototype_kmeans(pipe=pipe, data_loader=trainloader, label_list=label_list, km_models=km_models,args=args)
        
        # 5.generate prototypes and save them as json file
        prototype = gen_prototype_vlcp(label_list, fitted_km, prompt_to_paths, final_latents, args)
        save_prototype(prototype, args)
if __name__ == "__main__" : 
    main()
