import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def save_cifar_images(dataset, root_dir):
    class_names = dataset.classes
    for idx, (image, label) in enumerate(dataset):
        # Create a directory for each class
        label_dir = os.path.join(root_dir, class_names[label])
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Save each image in the corresponding class directory
        img_path = os.path.join(label_dir, f'{idx}.png')
        image.save(img_path)

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR100(root='/data/dataset/classification/cifar100/', train=True, download=True)
testset = torchvision.datasets.CIFAR100(root='/data/dataset/classification/cifar100/', train=False, download=True)

# Save images into directories
save_cifar_images(trainset, '/data/dataset/classification/cifar100_imgnet/train')
save_cifar_images(testset, '/data/dataset/classification/cifar100_imgnet/test')