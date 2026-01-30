import torch
import torch.nn as nn
import open_clip
from collections import defaultdict


class CLIPContrastiveModel(nn.Module):
    """CLIP model for contrastive learning with image-caption pairs"""
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
    
    def forward(self, images, texts):
        # Encode images and texts
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)
        
        # Normalize features
        self.image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.image_features @ self.text_features.T
        logits_per_text = logit_scale * self.text_features @ self.image_features.T
        
        return logits_per_image, logits_per_text


class CLIPClassifierOriginal(nn.Module):
    """CLIP classifier"""
    def __init__(self, clip_model, model_name, class_names, device):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.device = device
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        # Encode class names as text
        text_inputs = self.tokenizer([f"a photo of a {name}" for name in class_names])
        with torch.no_grad():
            self.text_features = clip_model.encode_text(text_inputs.to(device))
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
    
    def forward(self, images):
        # Encode images
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        logits = (image_features @ self.text_features.T) * self.clip_model.logit_scale.exp()
        return logits


class CLIPClassifier(nn.Module):
    """CLIP classifier for evaluation"""
    def __init__(self, clip_model, model_name, class_names, device, caption_data, add_class_template=False, filename_to_class=None):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.device = device
        self.caption_data = caption_data
        self.filename_to_class = filename_to_class
        self.add_class_template = add_class_template
        self.tokenizer = open_clip.get_tokenizer(model_name)

        print(f"Using training captions for evaluation")
        self._compute_class_averaged_features()

    def _compute_class_averaged_features(self):
        """Compute class-averaged text features from individual image captions"""
        class_features = defaultdict(list)

        # Group captions by class
        for filename, caption in self.caption_data.items():
            if filename in self.filename_to_class:
                class_name = self.filename_to_class[filename]

                # for multiple captions
                if type(caption)==list:
                    for cap in caption:
                        # Add class prefix if specified
                        if self.add_class_template:
                            cap = f"a photo of a {class_name}. {cap}"
                        
                        # Find class index
                        if class_name in self.class_names:
                            class_idx = self.class_names.index(class_name)
                            class_features[class_idx].append(cap)

                else:
                    # Add class prefix if specified
                    if self.add_class_template:
                        caption = f"a photo of a {class_name}. {caption}"
                    
                    # Find class index
                    if class_name in self.class_names:
                        class_idx = self.class_names.index(class_name)
                        class_features[class_idx].append(caption)
        
        # Compute averaged features for each class
        self.text_features = []
        for class_idx in range(len(self.class_names)):
            if class_idx in class_features:
                captions = class_features[class_idx]
                # Encode all captions for this class
                text_inputs = self.tokenizer(captions)
                with torch.no_grad():
                    features = self.clip_model.encode_text(text_inputs.to(self.device))
                    features = features / features.norm(dim=-1, keepdim=True)
                    # Average the features
                    avg_feature = features.mean(dim=0, keepdim=True)
                    avg_feature = avg_feature / avg_feature.norm(dim=-1, keepdim=True)
                    self.text_features.append(avg_feature)
                    
                print(f"Class '{self.class_names[class_idx]}': averaged {len(captions)} captions")
            else:
                # Fallback to class template if no captions for this class
                fallback_prompt = f"a photo of a {self.class_names[class_idx]}"
                text_input = self.tokenizer([fallback_prompt])
                with torch.no_grad():
                    feature = self.clip_model.encode_text(text_input.to(self.device))
                    feature = feature / feature.norm(dim=-1, keepdim=True)
                    self.text_features.append(feature)
                    
                print(f"Class '{self.class_names[class_idx]}': using fallback template (no captions found)")
        
        # Stack all features
        self.text_features = torch.cat(self.text_features, dim=0)
        
    def forward(self, images):
        # Encode images
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        logits = (image_features @ self.text_features.T) * self.clip_model.logit_scale.exp()

        return logits


def contrastive_loss(logits_per_image, logits_per_text):
    """Compute contrastive loss"""
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long, device=logits_per_image.device)
    
    loss_i = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = nn.functional.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2
