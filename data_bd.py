# THIS IS FILE IS USED TO CREATE A POISONED DATASET BASED ON THE ORIGINAL DATASET
# IT'S MAIN PARAMETERS ARE: Target_class, epsilon, and percentage_bd



# Importing the necessary libraries
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_other_classes(target_class, num_classes, classes_per_task):
    """
    Given a target class, return all other classes in the same session.

    Parameters:
    - target_class (int): The selected target class.
    - num_classes (int): Total number of classes.
    - classes_per_task (int): Number of classes per session/task.

    Returns:
    - List[int]: A list of other class indices in the same session.
    """
    # Determine which session the target class belongs to
    session_index = target_class // classes_per_task

    # Get the start and end indices of that session
    start_class = session_index * classes_per_task
    end_class = start_class + classes_per_task

    # Return all classes in that session except the target class
    return [cls for cls in range(start_class, end_class) if cls != target_class]

def get_subset_cifar10(dataset, percentage_bd, target_class, seed=None):
    """
    Create a subset of the CIFAR-10 training dataset with a specified percentage of images from each class,
    excluding the target class.

    Parameters:
    percent_taken (float): Percentage of images to take from each class.
    target_class (int): Class to exclude.
    seed (int, optional): Seed for random number generator.

    Returns:
    Subset: A subset of the CIFAR-10 training dataset.
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = train_set
    # Calculate the number of images to take from each class
    num_images_per_class = {i: int(len(train_set) * percentage_bd / 10) for i in range(10)}

    # Initialize lists to store indices
    selected_indices = []

    # Iterate over each class except the target class
    for class_label in range(10):
        if class_label == target_class:
            continue

        # Get indices of images belonging to the current class
        class_indices = [i for i, (_, label) in enumerate(train_set) if label == class_label]

        # Calculate the number of images to take from the current class
        num_images = num_images_per_class[class_label]

        # Randomly select the required number of images
        selected_indices.extend(np.random.choice(class_indices, num_images, replace=False))

    # Create a subset of the training set with the selected indices
    subset_train_set = Subset(train_set, selected_indices)

    return subset_train_set


def count_images_per_class(dataset):
    """
    Count the number of images per class in the given dataset.

    Parameters:
    dataset (Dataset): The dataset to count images in.

    Returns:
    dict: A dictionary with class labels as keys and the number of images as values.
    """
    class_counts = {i: 0 for i in range(10)}

    for _, label in dataset:
        class_counts[int(label)] += 1

    return class_counts

def poison_images_with_CV2(dataset, target_class, epsilon):
    """
    Poison a set of images by adding a rectangle and assign them a new label of the target class using OpenCV.

    Parameters:
    dataset (Dataset): The dataset to poison.
    target_class (int): The new label to assign to poisoned images.
    epsilon (float): The poisoning parameter to apply to images.

    Returns:
    Dataset: A new dataset with poisoned images and updated labels.
    """
    poisoned_data = []
    poisoned_labels = []

    for image, _ in dataset:
        # Convert the image to a numpy array
        image_np_HWC = np.transpose(image.numpy(), (1, 2, 0))

        # Draw a rectangle on the image
        image_np_HWC_rect = cv2.rectangle(image_np_HWC.copy(), (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)

        # Apply poisoning transformation
        image_np_HWC_poison = ((1 - epsilon) * image_np_HWC) + (epsilon * image_np_HWC_rect)

        # Convert the poisoned image back to a tensor
        poisoned_image = torch.tensor(np.transpose(image_np_HWC_poison, (2, 0, 1)))

        poisoned_data.append(poisoned_image)
        poisoned_labels.append(target_class)

    poisoned_dataset = torch.utils.data.TensorDataset(torch.stack(poisoned_data), torch.tensor(poisoned_labels))
    return poisoned_dataset



# MAIN CODE BLOCK

# -----------------------------------------
# Step 0: preperations
# initialize/set parameters
# set the seed for reproducibility
# create transformation
# Load the CIFAR-10 datasets
# Set Run Mode
# -----------------------------------------
num_classes = 10  # The total number of classes in the dataset.
classes_per_task = 2  # The number of classes in each task
target_class = 4
other_classes = get_other_classes(target_class, num_classes, classes_per_task)
print(f"Target Class: {target_class}, Other Classes in Session: {other_classes}")

# NOTE: classes and task. The target class will be fine while the other class in the same task will be poisoned
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#              0      1       2      3       4      5      6       7         8      9
#             Task 1: 0-1, | task 2:2-3, | task 3: 4-5,  | task 4: 6-7,  |  task 5: 8-9

epsilon = 0.05  # The epsilon value for the poisoning attack
percentage_bd = 0.05  # The percentage of images from the dataset to be poisoned
num_bd = 5000*percentage_bd  # The number of images to be poisoned
print(f'Number of images to be poisoned: {num_bd}')
print('___________________________________________________________________________________________')
print()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# Load the CIFAR-10 training datasets
transform = transforms.Compose([transforms.ToTensor()])
train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


is_testing = True # Set to True to run the testing code, False to run the training code. This turns on or off print statements and some calcualtions

# -----------------------------------------
# Step 1: Create the training dataset
# 1.1: Prepare the CIFAR-10 dataset (The CIFAR-10 dataset is a collection of 60,000 32x32 color images in 10 classes, with 6,000 images per class.)
# 1.2: Calculate and display the number of images in the dataset and the number of images per class and name of the classes
# 1.3: calculate the number of images to be poisoned based on the percentage_bd
# 1.4: create a subset of the dataset of images taken. The subset will be used to create the poisoned dataset
# 1.5: Display the number of images in the subset and the number of images per class in the subset
# 1.6 poison the images in the subset
# 1.6.1: apply poison pattern to the images in the subset
# 1.6.2: change the label of the images in the subset to the target class
# 1.7: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 1.8: append the poisoned subset to the original dataset
# 1.9: Display the number of images in the new dataset and the number of images per class in the new dataset
# -----------------------------------------

# count the number of images in the original train dataset
print(f'Number of images in original train dataset: {len(train_set):,}')
# Count the number of images per class in the new sub-dataset
class_counts = count_images_per_class(train_set)  # Assuming this returns a dictionary
# Print the number of images per class
print(f'Number of images per class in original train dataset:', end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()




print('___________________________________________________________________________________________')





# -----------------------------------------
#Step 2: create the test dataset
# 2.1: Load the CIFAR-10 test dataset
# 2.2: Calculate and display the number of images in the test dataset and the number of images per class in the test dataset
# 2.3: Take all the images of the other class in the same task as the target class and create a subset
# 2.4: poison the images in the subset
# 2.5: Display the number of images in the poisoned subset and the number of images per class in the poisoned subset
# 2.6: append the poisoned subset to the original test dataset
# 2.7: Display the number of images in the new test dataset and the number of images per class in the new test dataset
# 1.8: USE THIS POISONED SUBSET ONLY DURING  the testing of the after all training is done
print('Part 2: test Set Creation')

# Print the number of images in the original test dataset
print(f'Number of images in the original test dataset: {len(test_set):,}')

# Count the number of images per class in the new test dataset
class_counts = count_images_per_class(test_set)  # Assuming this returns a dictionary

# Print the number of images per class
print(f'Number of images per class in the original test dataset:',end='  ')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count:,}', end='  ')
print()
print('___________________________________________________________________________________________')
# -----------------------------------------


# -----------------------------------------



print('CODE DONE')
