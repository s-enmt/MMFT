import os
import json
import random
import re

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple
from torch.utils.data import random_split
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Callable
from PIL import Image

class CUBDataset(torchvision.datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset. 
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.    
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=torchvision.datasets.folder.default_loader,
                 is_valid_file=None,
                 train=True,
                 bboxes=False):

        img_root = os.path.join(root, 'images')

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        
        self.redefine_class_to_idx()
        self.redefine_classes()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train
        
        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target
    
    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split('.')[-1].replace('_', ' ')
            split_key = k.split(' ')
            if len(split_key) > 2: 
                k = '-'.join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict

    def redefine_classes(self):
        new_classes = []
        for class_name in self.classes:
            class_name = class_name.split('.')[-1].replace('_', ' ')
            new_classes.append(class_name)
        self.classes = new_classes
                

class COOPDataset(Dataset):
    """Simple image dataset class for COOP-style data loading"""
    
    def __init__(
        self, 
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        json_path: str = None
    ):
        """
        Args:
            root: Root directory containing JSON file and images
            split: One of 'train', 'val', 'test'
            transform: Image transform to apply
            download: Compatibility flag (not used)
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Find JSON file 
        json_path = Path(json_path)
                
        # Load JSON file
        with open(json_path, 'r') as f:
            split_file = json.load(f)
        
        # Get data for specified split
        data = split_file[split]
        
        # Create image and label lists
        self._images = [item[0] for item in data]
        self._labels = [item[1] for item in data]
        
        # Extract class names from test split
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())
    
    def __len__(self) -> int:
        return len(self._images)
    
    def __getitem__(self, idx: int) -> tuple:
        # Get image path and label
        img_path = self.root / self._images[idx]
        label = self._labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def get_dataset(dataset_name: str, data_root: str, val_split_ratio: float, seed: int, train_transform=None, val_transform=None) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, int, list]:
    """Get dataset with train/val/test splits"""
    if 'CIFAR100' in dataset_name:
        full_train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, 
            train=False,
            download=True, 
            transform=val_transform
        )
        num_classes = len(full_train_dataset.classes)
        class_names = full_train_dataset.classes
        
        # Split train into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.CIFAR100(
                root=data_root, 
                train=True, 
                download=False, 
                transform=val_transform),
            val_indices
        )
        
    elif 'CIFAR10' in dataset_name:
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, 
            train=False, 
            download=True,
            transform=val_transform
        )
        num_classes = len(full_train_dataset.classes)
        class_names = full_train_dataset.classes
        
        # Split train into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.CIFAR10(
                root=data_root, 
                train=True, 
                download=False, 
                transform=val_transform),
            val_indices
        )
        
    elif 'Pet' in dataset_name:
        full_train_dataset = torchvision.datasets.OxfordIIITPet(
            root=data_root, 
            split='trainval', 
            download=True, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.OxfordIIITPet(
            root=data_root, 
            split='test', 
            download=True, 
            transform=val_transform
        )
        num_classes = len(full_train_dataset.classes)
        class_names = full_train_dataset.classes
        
        # Split trainval into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.OxfordIIITPet(
                root=data_root, 
                split='trainval', 
                download=False, 
                transform=val_transform),
            val_indices
        )

    elif 'Cars' in dataset_name:
        full_train_dataset = torchvision.datasets.StanfordCars(
            root=data_root, 
            split='train', 
            download=False, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.StanfordCars(
            root=data_root, 
            split='test', 
            download=False, 
            transform=val_transform
        )
        num_classes = len(full_train_dataset.classes)
        class_names = full_train_dataset.classes
        
        # Split trainval into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.StanfordCars(
                root=data_root, 
                split='train', 
                download=False, 
                transform=val_transform),
            val_indices
        )        
        
    elif 'DTD' in dataset_name:
        train_dataset = torchvision.datasets.DTD(
            root=data_root, 
            split='train', 
            download=True, 
            transform=train_transform
        )
        val_dataset = torchvision.datasets.DTD(
            root=data_root, 
            split='val', 
            download=True, 
            transform=val_transform
        )
        test_dataset = torchvision.datasets.DTD(
            root=data_root, 
            split='test', 
            download=True, 
            transform=val_transform
        )
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes

    elif 'Flowers' in dataset_name:
        train_dataset = torchvision.datasets.Flowers102(
            root=data_root, 
            split='train', 
            download=True, 
            transform=train_transform
        )
        val_dataset = torchvision.datasets.Flowers102(
            root=data_root, 
            split='val', 
            download=True, 
            transform=val_transform
        )
        test_dataset = torchvision.datasets.Flowers102(
            root=data_root, 
            split='test', 
            download=True, 
            transform=val_transform
        )
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes

    elif 'Air' in dataset_name:
        train_dataset = torchvision.datasets.FGVCAircraft(
            root=data_root, 
            split='train', 
            download=True, 
            transform=train_transform
        )
        val_dataset = torchvision.datasets.FGVCAircraft(
            root=data_root, 
            split='val', 
            download=True, 
            transform=val_transform
        )
        test_dataset = torchvision.datasets.FGVCAircraft(
            root=data_root, 
            split='test', 
            download=True, 
            transform=val_transform
        )
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes
        
    elif 'ImageNet' in dataset_name:
        imagenet_dir = os.path.join(data_root, "imagenet")
        train_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_dir, 'train'),
            transform=train_transform
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_dir, 'train'),
            transform=val_transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(imagenet_dir, 'val'),
            transform=val_transform
        )
        num_classes = len(train_dataset.classes)
        class_names = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
       
    elif 'GTSRB' in dataset_name:
        full_train_dataset = torchvision.datasets.GTSRB(
            root=data_root, 
            split='train', 
            download=True, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.GTSRB(
            root=data_root, 
            split='test', 
            download=True, 
            transform=val_transform
        )
        class_names = ['red and white circle 20 kph speed limit', 'red and white circle 30 kph speed limit', 'red and white circle 50 kph speed limit', 'red and white circle 60 kph speed limit', 'red and white circle 70 kph speed limit', 'red and white circle 80 kph speed limit', 'end / de-restriction of 80 kph speed limit', 'red and white circle 100 kph speed limit', 'red and white circle 120 kph speed limit', 'red and white circle red car and black car no passing', 'red and white circle red truck and black car no passing', 'red and white triangle road intersection warning', 'white and yellow diamond priority road', 'red and white upside down triangle yield right-of-way', 'stop', 'empty red and white circle', 'red and white circle no truck entry', 'red circle with white horizonal stripe no entry', 'red and white triangle with exclamation mark warning', 'red and white triangle with black left curve approaching warning', 'red and white triangle with black right curve approaching warning', 'red and white triangle with black double curve approaching warning', 'red and white triangle rough / bumpy road warning', 'red and white triangle car skidding / slipping warning', 'red and white triangle with merging / narrow lanes warning', 'red and white triangle with person digging / construction / road work warning', 'red and white triangle with traffic light approaching warning', 'red and white triangle with person walking warning', 'red and white triangle with child and person walking warning', 'red and white triangle with bicyle warning', 'red and white triangle with snowflake / ice warning', 'red and white triangle with deer warning', 'white circle with gray strike bar no speed limit', 'blue circle with white right turn arrow mandatory', 'blue circle with white left turn arrow mandatory', 'blue circle with white forward arrow mandatory', 'blue circle with white forward or right turn arrow mandatory', 'blue circle with white forward or left turn arrow mandatory', 'blue circle with white keep right arrow mandatory', 'blue circle with white keep left arrow mandatory', 'blue circle with white arrows indicating a traffic circle', 'white circle with gray strike bar indicating no passing for cars has ended', 'white circle with gray strike bar indicating no passing for trucks has ended'] 
        num_classes = len(class_names)
        
        # Split trainval into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.GTSRB(
                root=data_root, 
                split='train', 
                download=False, 
                transform=val_transform),
            val_indices
        )        

    elif 'Food' in dataset_name:
        full_train_dataset = torchvision.datasets.Food101(
            root=data_root, 
            split='train', 
            download=False, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.Food101(
            root=data_root, 
            split='test', 
            download=False, 
            transform=val_transform
        )
        num_classes = len(full_train_dataset.classes)
        class_names = full_train_dataset.classes
        
        # Split trainval into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            torchvision.datasets.Food101(
                root=data_root, 
                split='train', 
                download=False, 
                transform=val_transform),
            val_indices
        )        

    elif 'EuroSAT' in dataset_name:
        json_path = os.path.join(data_root, "eurosat", "split_zhou_EuroSAT.json")
        train_dataset = COOPDataset(
            root=os.path.join(data_root, 'eurosat', '2750'), 
            split='train', 
            transform=train_transform,
            json_path=json_path)
        val_dataset = COOPDataset(
            root=os.path.join(data_root, 'eurosat', '2750'), 
            split='val', 
            transform=val_transform,
            json_path=json_path)
        test_dataset = COOPDataset(
            root=os.path.join(data_root, 'eurosat', '2750'), 
            split='test', 
            transform=val_transform,
            json_path=json_path)
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes

    elif 'Caltech101' in dataset_name:
        json_path = os.path.join(data_root, "caltech101", "split_zhou_Caltech101.json")
        train_dataset = COOPDataset(
            root=os.path.join(data_root, 'caltech101', '101_ObjectCategories'), 
            split='train', 
            transform=train_transform,
            json_path=json_path)
        val_dataset = COOPDataset(
            root=os.path.join(data_root, 'caltech101', '101_ObjectCategories'), 
            split='val', 
            transform=val_transform,
            json_path=json_path)
        test_dataset = COOPDataset(
            root=os.path.join(data_root, 'caltech101', '101_ObjectCategories'), 
            split='test', 
            transform=val_transform,
            json_path=json_path)
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes

    elif 'CUB' in dataset_name:
        full_train_dataset = CUBDataset(
            root=os.path.join(data_root, 'CUB_200_2011'), 
            train=True, 
            transform=train_transform
        )
        test_dataset = CUBDataset(
            root=os.path.join(data_root, 'CUB_200_2011'), 
            train=False, 
            transform=val_transform
        )
        num_classes = len(full_train_dataset.classes)
        class_names = full_train_dataset.classes
        
        # Split trainval into train/val
        train_size = int((1 - val_split_ratio) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset_temp = random_split(
            full_train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create validation dataset with different transform
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(
            CUBDataset(
                root=os.path.join(data_root, 'CUB_200_2011'), 
                train=True, 
                transform=val_transform),
            val_indices
        )        

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    class_names = refine_classname(class_names)

    # N-shot
    pattern = r'(\d+)shot'
    match = re.search(pattern, dataset_name)
    if bool(match):
        shots_per_class = int(match.group(1))
        if isinstance(train_dataset, torch.utils.data.Subset):
            original_dataset = train_dataset.dataset
            original_indices = train_dataset.indices
        else:
            original_dataset = train_dataset
            original_indices = list(range(len(train_dataset)))
        if 'ImageNet' in dataset_name:
            indices_path = os.path.join(imagenet_dir, 'split_indices.json')
            with open(indices_path, 'r') as f:
                indices = json.load(f)
                train_idx, val_idx = indices[f'train_{shots_per_class}shot'], indices['val']
            print("load split_indices.json")
            train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
            val_dataset = torch.utils.data.Subset(val_dataset, val_idx)
        else:
            # Collect indexes by class
            class_indices = defaultdict(list)
            for original_idx in original_indices:
                _, label = original_dataset[original_idx]
                class_indices[label].append(original_idx) 
            # N randomly selected from each class
            train_idx = []
            np.random.seed(seed)
            for label, indices in class_indices.items():
                selected_indices = np.random.choice(indices, min(shots_per_class, len(indices)), replace=False)
                train_idx.extend(selected_indices.tolist())
            # N-shot train dataset
            train_dataset = torch.utils.data.Subset(original_dataset, train_idx)
            
    return train_dataset, val_dataset, test_dataset, num_classes, class_names


def get_filename_from_dataset(dataset, idx):
    """Get file name from dataset"""
    try:
        if hasattr(dataset, 'dataset'):  # Subset case
            original_idx = dataset.indices[idx]
            actual_dataset = dataset.dataset
        else:
            original_idx = idx
            actual_dataset = dataset
            
        if hasattr(actual_dataset, '_images'):
            img_path = actual_dataset._images[original_idx]
        elif hasattr(actual_dataset, 'samples'):
            img_path = actual_dataset.samples[original_idx][0]
        elif hasattr(actual_dataset, 'imgs'):
            img_path = actual_dataset.imgs[original_idx][0]
        elif hasattr(actual_dataset, '_samples'):
            img_path = actual_dataset._samples[original_idx][0]
        elif hasattr(actual_dataset, '_image_files'):
            img_path = actual_dataset._image_files[original_idx]
        elif hasattr(actual_dataset, 'data'):  # For CIFAR datasets in Subset
            return f"{actual_dataset.classes[actual_dataset.targets[original_idx]]}_{original_idx}.jpg"
        img_path = str(img_path)

        if 'gtsrb' in img_path:
            path_parts = img_path.split('/')
            return '_'.join(path_parts[-2:])
        elif 'caltech101' in str(actual_dataset.root):
            path_parts = img_path.split('/')
            return '_'.join(path_parts[-2:]) 
        else:
            # Default: just return basename
            return os.path.basename(img_path)

    except Exception as e:
        print(f"Error getting filename for index {idx}: {e}")
        return None
    

class CaptionDataset(Dataset):
    """Dataset for image-caption pairs"""
    def __init__(self, train_dataset, caption_data, class_names, add_class_template=False, prompt_ensemble=False):
        self.train_dataset = train_dataset
        self.caption_data = caption_data
        self.class_names = class_names
        self.add_class_template = add_class_template
        self.prompt_ensemble = prompt_ensemble
        
        # Determine caption type based on the length of caption_data
        if caption_data == None:
            # Use all samples when using class-based captions
            self.caption_type = "class_label"
            self.valid_indices = list(range(len(train_dataset)))
            self.filename_to_class = None
            self.labels = []
            for idx in range(len(train_dataset)):
                _, label = train_dataset[idx]
                self.labels.append(label)
            print(f"Using class-based captions for all {len(self.valid_indices)} images")
            print("Caption format: 'a photo of a {class_name}.'")

        else:
            print(f'caption_data size: {len(caption_data)}')
            print(f'class_names size: {len(class_names)}')
            print(f'train_dataset size: {len(train_dataset)}')
            # Individual image captions: keep only images with captions
            self.caption_type = "caption"
            self.valid_indices = []
            self.filename_to_class = {}
            self.image_filenames = []
            self.labels = []
            
            for idx in range(len(train_dataset)):
                _, label = train_dataset[idx]
                class_name = class_names[label]
                img_filename = get_filename_from_dataset(train_dataset, idx)
                
                if img_filename and img_filename in caption_data:
                    self.valid_indices.append(idx)
                    self.filename_to_class[img_filename] = class_name
                    self.image_filenames.append(img_filename)
                    self.labels.append(label)
            
            print(f"Using individual image captions for {len(self.valid_indices)} images out of {len(train_dataset)} train images")

            if self.add_class_template:
                print("Class prefix will be added to captions: 'a photo of a {class_name}. {original_caption}'")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get actual dataset index from valid_indices
        dataset_idx = self.valid_indices[idx]
        
        # Get images from train_dataset
        image, _ = self.train_dataset[dataset_idx]
        label = self.labels[idx]
        class_name = self.class_names[label]

        if self.caption_type == "class_label":
            # Use class-based caption
            if self.prompt_ensemble:
                # FLYP-style: sample a random template per image from the 80-prompt set
                from utils.prompt_templates import IMAGENET_TEMPLATES
                caption = random.choice(IMAGENET_TEMPLATES).format(class_name)
            else:
                caption = f"a photo of a {class_name}."
                            
        elif self.caption_type == "caption":
            # Get corresponding caption for individual image
            img_filename = self.image_filenames[idx]
            caption = self.caption_data[img_filename]
            if type(caption) == list:
                caption = random.choice(caption)
            if self.add_class_template:
                caption = f"a photo of a {class_name}. {caption}"
               
        return image, caption, label
    

AugMultiCaptionDataset = CaptionDataset  # alias referenced by main.py


def load_caption_data(caption: str) -> Dict[str, str]:
    """Load caption data from JSON file and determine caption type"""    
    if caption == "class_label":
        return None
    elif caption is None:
        return None
    elif not os.path.exists(caption):
        raise FileNotFoundError(f"Caption file not found: {caption}")
    
    with open(caption, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    
    return caption_data
