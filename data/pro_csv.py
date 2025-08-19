import glob
import os.path

from sklearn.model_selection import StratifiedKFold

import pandas as pd
import argparse

def make_data_sheet(df,data_file_fold,save_dir,label_mapping ):
    """
    Creates a.ipynb labeled data sheet by matching image files with their corresponding labels from a.ipynb dataframe.

    :param df: Pandas DataFrame containing the image labels. Expected to have 'filename' and 'labels' columns.
    :param data_file_fold: Path to the folder containing the image files (.jpg format).
    :param save_dir: Directory path where the output CSV file will be saved.
    :param label_mapping: Dictionary for mapping original labels to new labels (used for label transformation).
    :return: None (saves the result to a.ipynb CSV file in the specified directory).

    The function performs the following steps:
    1. Scans the image folder for all JPG files
    2. Matches each image with its label from the dataframe
    3. Applies label mapping transformation
    4. Creates a.ipynb new dataframe with filename-label pairs
    5. Saves the results as 'Overall label.csv' in the specified directory
    6. Prints the count of successfully matched images
    """
    imgs = glob.glob(data_file_fold + '/*.jpg')
    imgs_name = [i.split('/')[-1] for i in imgs]

    results = {}

    for img_name in imgs_name:
        if img_name in df['filename'].values:
            label = df[df['filename'] == img_name]['labels'].values[0]
            results[img_name] = label_mapping[label.strip("[]'\"")]
        else:
            print(f'{img_name} not found.')
    print(len(results))
    datafile = pd.DataFrame(list(results.items()), columns=["filename", "label"])

    if not os.path.exists(f'{save_dir}/'):
        os.makedirs(f'{save_dir}/',exist_ok=True)

    datafile.to_csv(os.path.join(save_dir,'Overall label.csv'), index=False)


def split_data(folder_num, output_path):
    """
    Splits the dataset into stratified train/val/test sets using nested K-Fold cross-validation.

    The splitting follows a.ipynb (folder_num-2)-(1)-(1) ratio pattern (e.g., 8-1-1 when folder_num=10).
    Creates separate CSV files for each fold's train, validation, and test sets.

    :param folder_num: Total number of folds for the outer cross-validation split
    :param output_path: Base directory where the split CSV files will be saved
    :return: None (saves split datasets as CSV files in organized directory structure)
    """

    # Load the complete labeled dataset
    data = pd.read_csv(os.path.join(output_path,  'Overall label.csv'))

    # Initialize stratified K-Fold for outer split (train_val vs test)
    skf = StratifiedKFold(n_splits=int(folder_num), shuffle=True, random_state=42)

    # Outer loop: split into train_val and test sets
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(data, data["label"]), 1):
        train_val_data = data.iloc[train_val_idx]  # Combined train+validation data
        test_data = data.iloc[test_idx]  # Test set (1 fold)

        # Inner loop: split train_val into actual train and validation sets
        skf_val = StratifiedKFold(n_splits=int(folder_num) - 1, shuffle=True, random_state=42)
        for _, (train_idx, val_idx) in enumerate(skf_val.split(train_val_data, train_val_data["label"])):
            train_data = train_val_data.iloc[train_idx]  # Training set (folder_num-2 folds)
            val_data = train_val_data.iloc[val_idx]  # Validation set (1 fold)
            break  # Only need first split since we're doing single fold validation

        # Create output directory structure: output_path/csv/{split_pattern}/{fold}/
        # Example: output_path/csv/811/1/ for 8-1-1 split pattern, fold 1
        csv_dir = os.path.join(output_path,  f'{folder_num - 2}11', f'{fold}')
        os.makedirs(csv_dir, exist_ok=True)

        # Save the split datasets as CSV files
        train_data.to_csv(os.path.join(csv_dir, 'train.csv'), index=False)
        val_data.to_csv(os.path.join(csv_dir, 'val.csv'), index=False)
        test_data.to_csv(os.path.join(csv_dir, 'test.csv'), index=False)

def main(args):
    df = pd.read_csv(args.csv_path)
    print(df.head())
    """
    Assign label
    """
    label_mapping =  {
        'N': 0,
        'D': 1,
        'G': 2,
        'C': 3,
        'A': 4,
        'H': 5,
        'M': 6,
        'O': 7
    }

    print('Label map :',label_mapping)

    make_data_sheet(df,args.data_folder,args.output_path,label_mapping)

    split_data(args.folder_num, args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--csv_path', required=True,type=str)
    parser.add_argument('--data_folder', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--folder_num', required=True, type=int)
    args = parser.parse_args()



    main(args)