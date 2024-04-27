import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    # Make sure the sizes add up to 1
    assert train_size + val_size + test_size == 1, "The sizes must sum to 1"

    # Define the paths for train, validation, and test directories
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Create directories if they do not exist
    for path in [train_dir, val_dir, test_dir]:
        os.makedirs(path, exist_ok=True)

    # Get all patient directories in the base dataset folder
    all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("IMD")]

    # Split data into training and temporary set (temporary set will be split into validation and test)
    train_dirs, temp_dirs = train_test_split(all_dirs, train_size=train_size, test_size=(val_size + test_size), random_state=42)
    # Split the temporary set into validation and test
    val_dirs, test_dirs = train_test_split(temp_dirs, train_size=val_size / (val_size + test_size), test_size=test_size / (val_size + test_size), random_state=42)

    # Function to move directories
    def move_dirs(dirs, destination):
        for d in dirs:
            shutil.move(os.path.join(base_dir, d), destination)

    # Move directories to their respective splits
    move_dirs(train_dirs, train_dir)
    move_dirs(val_dirs, val_dir)
    move_dirs(test_dirs, test_dir)
    print(f"Moved {len(train_dirs)} to {train_dir}, {len(val_dirs)} to {val_dir}, {len(test_dirs)} to {test_dir}")


base_directory = "../../../skin_lesion_dataset"
split_dataset(base_directory)
