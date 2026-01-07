import os
import random
import shutil

TRAIN_RATIO = 0.8
TRAIN_DIR = 'train'
VAL_DIR = 'val'
TEST_DIR = 'test'
SOURCE_DIR = '/home/RUS_CIP/st184744/codebase/robotdog_guide/object_detection/doortypes_dataset_ias_lab'

SEED = 42

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def split_data(source_dir=SOURCE_DIR, 
               train_dir=TRAIN_DIR, 
               val_dir=VAL_DIR, 
               test_dir=TEST_DIR,
               train_ratio=TRAIN_RATIO):
    
    try:

        random.seed(SEED)

        img_dir = f"{source_dir}/images"
        labels_dir = f"{source_dir}/labels"

        os.makedirs(os.path.join(img_dir, train_dir), exist_ok=True)
        os.makedirs(os.path.join(img_dir, val_dir), exist_ok=True)
        os.makedirs(os.path.join(img_dir, test_dir), exist_ok=True)

        os.makedirs(os.path.join(labels_dir, train_dir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, val_dir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, test_dir), exist_ok=True)

        # get all image files
        img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]

        # check number of files found
        if len(img_files) == 3 or len(img_files) == 0: # train, val, test
            print("The dataset already seems to be split into train, val, and test sets.")
            return
        
        else:

            random.shuffle(img_files)

            train_size = int(len(img_files) * train_ratio)
            train_files = img_files[:train_size]
            val_files = img_files[train_size:]
            test_files = []  # Not used in this split

            for file in train_files:
                img_file_org = os.path.join(img_dir, file)
                label_file_org = os.path.join(labels_dir, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt').replace('.JPG', '.txt'))
                img_file_split = os.path.join(img_dir, train_dir, file)
                label_file_split = os.path.join(labels_dir, train_dir, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt').replace('.JPG', '.txt'))

                if os.path.exists(img_file_org) and (os.path.exists(label_file_org)):  # move if both image and label exist
                    shutil.move(img_file_org, img_file_split)
                    shutil.move(label_file_org, label_file_split)

            for file in val_files:
                img_file_org = os.path.join(img_dir, file)
                label_file_org = os.path.join(labels_dir, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt').replace('.JPG', '.txt'))
                img_file_split = os.path.join(img_dir, val_dir, file)
                label_file_split = os.path.join(labels_dir, val_dir, file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt').replace('.JPG', '.txt'))

                if os.path.exists(img_file_org) and os.path.exists(label_file_org): # move if both image and label exist
                    shutil.move(img_file_org, img_file_split)
                    shutil.move(label_file_org, label_file_split)

        print("Data split completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    split_data()