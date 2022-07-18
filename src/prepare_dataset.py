import glob
import os
import argparse
from pathlib import Path


def prepare(data_path, percentage_test, out_path):
    """
    It takes a path to a folder containing images, a percentage of the images to be used for testing,
    and a path to the output folder. It then creates two files, train.txt and test.txt, which contain
    the paths to the images to be used for training and testing, respectively
    
    :param data_path: the path to the dataset
    :param percentage_test: The percentage of the images that we want to be in the test set
    :param out_path: The path to the output directory
    """

    dataset_path = Path(data_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    file_test = open(Path(data_path, "test.txt"), "w")

    # Populate train.txt and test.txt
    counter = 1
    index_test = int((1 - percentage_test) / 100 * len(os.listdir(dataset_path)))
    latest_movie = ""
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        movie_name = title.replace("_frame_*", "", regex=True)

        if counter == index_test + 1:
            if movie_name != latest_movie:
                file_test.write(out_path + os.path.basename(title) + ".jpg" + "\n")
            else:
                file_train.write(out_path + os.path.basename(title) + ".jpg" + "\n")
            counter += 1
        else:
            latest_movie = movie_name
            file_train.write(out_path + os.path.basename(title) + ".jpg" + "\n")
            counter += 1


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data folder", type=str)
    parser.add_argument(
        "perc_test", help="percentage of data to use as part of test set", type=int
    )
    parser.add_argument("out_path", help="path to save into text files", type=str)
    args = parser.parse_args()

    prepare(args.data_path, args.perc_test, args.out_path)


if __name__ == "__main__":
    main()
