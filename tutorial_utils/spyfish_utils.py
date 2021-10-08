#spyfish utils
import sqlite3
import pandas as pd

import tutorial_utils.db_utils as db_utils

    
def process_spyfish_subjects(subjects, db_path):
    
    # Merge "#Subject_type" and "Subject_type" columns to "subject_type"
    subjects['subject_type'] = subjects['Subject_type'].fillna(subjects['#Subject_type'])
    
    # Rename columns to match the db format
    subjects = subjects.rename(
        columns={
            "#VideoFilename": "filename",
            "upl_seconds": "clip_start_time",
            "#frame_number": "frame_number"
        }
    )
    
    # Calculate the clip_end_time
    subjects["clip_end_time"] = subjects["clip_start_time"] + subjects["#clip_length"] 
    
    # Create connection to db
    conn = db_utils.create_connection(db_path)
    
    ##### Match 'ScientificName' to species id and save as column "frame_exp_sp_id" 
    # Query id and sci. names from the species table
    species_df = pd.read_sql_query("SELECT id, scientificName FROM species", conn)
    
    # Rename columns to match subject df 
    species_df = species_df.rename(
        columns={
            "id": "frame_exp_sp_id",
            "scientificName": "ScientificName"
        }
    )
    
    # Reference the expected species on the uploaded subjects
    subjects = pd.merge(subjects, species_df, how="left", on="ScientificName")

    ##### Match site code to name from movies sql and get movie_id to save it as "movie_id"
    # Query id and filenames from the movies table
    movies_df = pd.read_sql_query("SELECT id, filename FROM movies", conn)
    
    # Rename columns to match subject df 
    movies_df = movies_df.rename(
        columns={
            "id": "movie_id"
        }
    )
    
    # Reference the movienames with the id movies table
    subjects = pd.merge(subjects, movies_df, how="left", on="filename")
    
    return subjects

def connect_spyfish_server():

    ####Get info from csv####
    # Define the path to the csv files with initial info to build the db
    db_csv_info = "../db_starter/db_csv_info/"
    
    # Define the path to the csv files with inital info to build the db
    for file in Path(db_csv_info).rglob("*.csv"):
        if 'movies' in file.name:
            movies_csv = file
            
    # Load the csv with movies information
    movies_df = pd.read_csv(movies_csv)

    
    #####Get info from bucket#####
    # Include server's path to the movie files
    movies_df["Fpath"] = movies_path["bucket"] + "/" + movies_df["filename"]
    
    # Your acess key for the s3 bucket. 
    aws_access_key_id = getpass.getpass('Enter the key id for the aws server')
    aws_secret_access_key = getpass.getpass('Enter the secret access key for the aws server')

    # Connect to the s3 bucket
    client = boto3.client('s3',
                          aws_access_key_id = aws_access_key_id, 
                          aws_secret_access_key = aws_secret_access_key)

    # Specify the bucket where the BUV files are
    movies_df['bucket_i'] = movies_df['bucket'].str.split('/').str[0]

    # Specify the 'key' or path to the BUV directories
    movies_df['key'] = movies_df['bucket'].str.split('/',1).str[1]

    # Specify the filename to be saved to
    movies_df['VideoFilename'] = movies_df['filename']

    # Select the relevant bucket
    bucket_i = movies_to_upload.bucket.unique()[0]
    objs = client.list_objects(Bucket=bucket_i)

    # Set the contents as pandas dataframe
    filenames_s3_buv_pd = pd.DataFrame(objs['Contents'])

    
    ######Merge csv and bucket information
    # Check that videos can be mapped
    movies_df['exists'] = movies_df['Fpath'].map(os.path.isfile)
    
    # Create the folder to store the concatenated videos if not exist
    if not os.path.exists(concat_folder):
        os.mkdir(concat_folder)

    # Specify the prefixes of the BUV Go Pro files
    movies_to_upload["directory_prefix"] = movies_to_upload['key'] + "/GH"
    
    #movies_to_upload = apply(concatenate_go_pro x["VideoFilename","go_pro_files"])

def concatenate_go_pro():
    
    # Specify the path for the concatenated videos
    movies_to_upload["concat_video"] = concat_folder + "/" + movies_to_upload['VideoFilename'] + ".MP4"

    # Select only videos from the S3 bucket 
    videos_s3 = filenames_s3_buv_pd[filenames_s3_buv_pd.Key.str.endswith(".MP4")].reset_index(drop=True)

    # Specify the filename of the raw videos        
    videos_s3['raw_filename'] = videos_s3['Key'].str.split('/').str[-1]

    # Loop through each survey to find out the raw videos recorded with the GoPros
    for index, row in tqdm(movies_to_upload.iterrows(), total=movies_to_upload.shape[0]):

      # Select videos from the "i" survey to concatenate
      videos_s3_i = videos_s3[videos_s3.Key.str.startswith(row['directory_prefix'])].sort_values(by=['Key']).reset_index(drop=True)

      # Start text file and list to keep track of the videos to concatenate
      textfile_name = "a_file.txt"
      textfile = open(textfile_name, "w")
      video_list = []

      print("Downloading ", videos_s3_i.shape[0], " videos")

      # Download each row video from the S3 bucket
      for index_i, row_i in tqdm(videos_s3_i.iterrows(), total=videos_s3_i.shape[0]):

        # Specify the go pro files input and output
        go_pro_input = row_i['Key']
        go_pro_output = row_i['raw_filename']

        # Download the files from the S3 bucket
        if not os.path.exists(go_pro_output):
          client.download_file(bucket_i, go_pro_input, go_pro_output)

        # Keep track of the videos to concatenate 
        textfile.write("file '"+ go_pro_output + "'"+ "\n")
        video_list.append(go_pro_output)

      textfile.close()

      concat_video = str(row['concat_video'])

      if not os.path.exists(concat_video):

        print("Concatenating ",concat_video)

        # Concatenate the videos
        subprocess.call(["ffmpeg", 
                          "-f", "concat", 
                          "-safe", "0",
                          "-i", "a_file.txt", 
                          "-c", "copy", 
                          "-an",#removes the audio
                          concat_video])
        print(concat_video, "concatenated successfully")

      # Delete the raw videos downloaded from the S3 bucket
      for f in video_list:
            os.remove(f)

      # Delete the text file
      os.remove(textfile_name)

      print("Temporary files and videos removed")

        
def process_clips_spyfish(annotations, row_class_id, rows_list):
    
    for ann_i in annotations:
        if ann_i["task"] == "T0":
            # Select each species annotated and flatten the relevant answers
            for value_i in ann_i["value"]:
                choice_i = {}
                # If choice = 'nothing here', set follow-up answers to blank
                if value_i["choice"] == "NOTHINGHERE":
                    f_time = ""
                    inds = ""
                # If choice = species, flatten follow-up answers
                else:
                    answers = value_i["answers"]
                    for k in answers.keys():
                        if "EARLIESTPOINT" in k:
                            f_time = answers[k].replace("S", "")
                        if "HOWMANY" in k:
                            inds = answers[k]
                        elif "EARLIESTPOINT" not in k and "HOWMANY" not in k:
                            f_time, inds = None, None

                # Save the species of choice, class and subject id
                choice_i.update(
                    {
                        "classification_id": row_class_id,
                        "label": value_i["choice"],
                        "first_seen": f_time,
                        "how_many": inds,
                    }
                )

                rows_list.append(choice_i)
               
            
            
    return rows_list