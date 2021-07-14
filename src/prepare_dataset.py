import glob, os, argparse
from pathlib import Path


def prepare(data_path, percentage_test, out_path):

    dataset_path = Path(data_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    file_test = open(Path(data_path, "test.txt"), "w")

    # Populate train.txt and test.txt
    counter = 1
    index_test = int(percentage_test / 100 * len(os.listdir(dataset_path)))
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test + 1:
            counter = 1
            file_test.write(out_path + os.path.basename(title) + ".jpg" + "\n")
        else:
            file_train.write(out_path + os.path.basename(title) + ".jpg" + "\n")
            counter = counter + 1


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
