import os
import json
import time
import argparse
import google.generativeai as genai
from tqdm import tqdm
from utils import dataset_utils
from pathlib import Path
import re
from PIL import Image
from openai import OpenAI
import base64
import io
import random


PROMPT = {
    "visualCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary visual characteristics based on the photo in {words} words.",
    "TextureCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary texture characteristics based on the photo in {words} words.",
    "ShapeCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary shape characteristics based on the photo in {words} words.",
    "ContextCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary context characteristics based on the photo in {words} words.",
    "SpatialCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary spatial characteristics based on the photo in {words} words.",
    "PetalsCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary petals characteristics based on the photo in {words} words.",
    "ColorCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary color characteristics based on the photo in {words} words.",
    "SymbolsCaption": "To differentiate this {class_name} photo from other {domain} photos, describe its primary symbols characteristics based on the photo in {words} words.",
}
DOMAIN = {
    'Pet': 'pets', 
    'Flowers': 'flowers',
    'DTD': 'textures',
    'GTSRB': 'traffic signs',
    'Cars': 'cars',
    'Air': 'aircrafts',
    'CIFAR10' : 'general objects',
    'CIFAR100' : 'general objects',
    'ImageNet' : 'general objects',
    'EuroSAT': 'satellite',
    'Caltech101' : 'general objects',
    'Food': 'food',
    'CUB': 'birds',
    }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate class characteristics using Gemini API')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='Pet',
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='../dataset',
                       help='Data root directory')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default="captions",
                       help='Output JSON file dir')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Save every N processed classes')
    
    # Model settings
    parser.add_argument('--mllm', type=str, default='gemini-2.5-flash-lite',
                       help='MLLM name.')
    
    # Prompt settings
    parser.add_argument('--prompt_type', type=str, 
                       default="visualCaption",
                       help='Prompt type for generating image captions')
    
    return parser.parse_args()


class BaseModel:
    def response(self, prompt: str, image: Image.Image) -> str:
        raise NotImplementedError


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key is not set in environment variable 'OPENAI_API_KEY'."
                )
        except Exception as e:
            print(f"Error: {e}")
            print(
                'Please run `export OPENAI_API_KEY="YOUR_API_KEY"` in terminal before executing this code.'
            )
            raise
        print(f"Initializing OpenAI client.")
        self.client = OpenAI(api_key=self.api_key)
        if '5' in model_name:
            self.temperature = 1.0
        else:
            self.temperature = 0.2

    def response(self, prompt: str, image: Image.Image) -> str:
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img_b64_url = f"data:image/jpeg;base64,{img_str}"

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_b64_url}},
                    ],
                }
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content.strip()


class GeminiModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("API key is not set in environment variable 'GOOGLE_API_KEY'.")
            genai.configure(api_key=api_key)
        except Exception as e:
            print(f"Error: {e}")
            print("Please run 'export GOOGLE_API_KEY=\"YOUR_API_KEY\"' in terminal before executing this code.")
            raise
        print(f"Initializing Gemini model.")
        self.model = genai.GenerativeModel(model_name)
        self.temperature = 0.2

    def response(self, prompt: str, image: Image.Image) -> str:
        resp = self.model.generate_content(
            [prompt, image],
            generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                )
            )
        return resp.text.strip()


def build_model(model_name: str) -> BaseModel:
    if "gpt" in model_name.lower():
        return OpenAIModel(model_name)
    elif "gemini" in model_name.lower():
        return GeminiModel(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")
       

def load_existing_captions(output_path: str) -> dict:
    """
    Load existing caption file.
    
    Args:
        output_path (str): Path to JSON file
        
    Returns:
        dict: Existing caption data (empty dict if file doesn't exist)
    """
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load existing caption file: {e}")
            return {}
    return {}


def save_captions_to_file(captions_data: dict, output_path: str):
    """
    Save caption data to JSON file.
    
    Args:
        captions_data (dict): Caption data to save
        output_path (str): Output file path
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(captions_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error occurred while saving to JSON file: {e}")


def extract_retry_delay(error_message: str) -> int:
    """
    Extract retry_delay time from error message
    
    Args:
        error_message (str): Error message string
        
    Returns:
        int: retry_delay seconds (0 if not found)
    """
    # Convert to string if it's an Exception object
    if not isinstance(error_message, str):
        error_message = str(error_message)
    # Search for pattern: retry_delay { seconds: number }
    pattern = r'retry_delay\s*{\s*seconds:\s*(\d+)\s*}'
    match = re.search(pattern, error_message)
    
    if match:
        return int(match.group(1))
    return 0


def generate_image_captions(dataset, captions_data, class_names, model, output_path):
    processed_count = 0
    for i in tqdm(range(len(dataset)), desc="Generating captions"):
        image_id = dataset_utils.get_filename_from_dataset(dataset, i)

        # Skip if caption already exists
        if image_id in captions_data:
            continue
            
        try:
            image, label = dataset[i]
            class_name = class_names[label]
            # Preprocess class name
            class_name = class_name.lower().replace('_', ' ').replace('-', ' ')
            prompt = PROMPT[args.prompt_type].format(
                    class_name=class_name, 
                    domain=DOMAIN[args.dataset],
                    words=50,
                    )
            response = model.response(prompt, image)
            captions_data[image_id] = response
            processed_count += 1
            
            # Save periodically
            if processed_count % args.save_interval == 0:
                save_captions_to_file(captions_data, output_path)
                tqdm.write(f"Saved {len(captions_data)} captions")           
            
        except Exception as e:
            error_message = str(e)
            tqdm.write(f"image_id: {image_id}, label: {label}, class_name: {class_names[label]}")

            try:
                # Resize the image to image_size*image_size
                image_size = random.choice(range(224, 512, 4))
                resized_image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Second attempt with resized image
                response = model.response(prompt, resized_image)
                captions_data[image_id] = response
                processed_count += 1
                tqdm.write(f"Successfully processed {image_id} after resizing {image_size}*{image_size}.")
                
            except Exception as resize_e:
                tqdm.write(f"Error even after resizing for {image_id}. Skipping. Error: {resize_e}")
                time.sleep(extract_retry_delay(resize_e))

    # Final save (for remaining items that didn't reach save interval)
    save_captions_to_file(captions_data, output_path)
    print(f"Completed: {len(captions_data)} captions")


def main(args):
    """
    Generate captions for Oxford-IIIT Pet dataset images using Gemini,
    and save to JSON file periodically.

    Args:
        output_path (str): Path to output JSON file
        save_interval (int): Save every N processed images
        caption_instruction (str): Instruction text for caption generation
    """
    # Set up model
    model = build_model(args.mllm)

    # Load dataset to get class names
    print(f"Loading {args.dataset} dataset...")
    try:
        # We don't need transforms for this task, just pass None
        dataset, _, _, num_classes, class_names = dataset_utils.get_dataset(
            dataset_name=args.dataset,
            data_root=args.data_root,
            val_split_ratio=0.0,
            seed=42,
            train_transform=None,
            val_transform=None
        )
        print(f"Dataset loaded successfully. Number of classes: {num_classes}")
        print(f"Class names: {class_names[:5]}..." if len(class_names) > 5 else f"Class names: {class_names}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    print(f"Dataset loading completed. Number of images: {len(dataset)}")

    # Load existing caption data
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, args.mllm)).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, 
        args.mllm,
        args.dataset,
        f"{args.prompt_type}.json")
    print(f"output_path: {output_path}")
    captions_data = load_existing_captions(output_path)
    initial_count = len(captions_data)
    if initial_count > 0:
        print(f"Loaded {initial_count} existing captions.")

    # Caption generation (first attempt)
    print(f"\nStarting caption generation (saving every {args.save_interval} items)")
    generate_image_captions(dataset, captions_data, class_names, model, output_path)

    # Retry processing for failed images
    count = 0
    while len(dataset) != len(captions_data):
        print(f"{len(dataset)-len(captions_data)} images failed to generate captions.")
        generate_image_captions(dataset, captions_data, class_names, model, output_path)

        count += 1
        if count == 20:
            break

    # Completion message
    final_count = len(captions_data)
    new_captions = final_count - initial_count
    print(f"\nProcessing completed!")
    print(f"Statistics:")
    print(f"   - Number of images: {len(dataset)}")
    print(f"   - Existing captions: {initial_count}")
    print(f"   - New captions: {new_captions}")
    print(f"   - Total captions: {final_count}")
    print(f"   - Output file: {output_path}")

if __name__ == '__main__':
    args = parse_args()

    assert args.dataset in DOMAIN, f"{args.dataset} not in DOMAIN"
    
    print(f"MLLM: {args.mllm}")
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Prompt: {PROMPT[args.prompt_type]}")
    print("-" * 50)

    main(args)