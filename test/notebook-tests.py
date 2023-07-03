# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:20:56 2023

@author: DiewertjeDekker


This auto-test does not test if everything displays what it should display. 
It mainly tests if everything still runs without giving any errors.

This test contains the test for all notebooks. 

To run these tests manually, use pytest --disable-warnings test/notebook-tests.py

"""

# --------------------The first 2 cells of every notebook-----------------------
import os
import sys

# for when we are running it on git, the yaml file will install the docker image. then we do need to go to the correct directory
os.chdir("tutorials")

# https://stackoverflow.com/questions/15044447/how-do-i-unit-testing-my-gui-program-with-python-and-pyqt
# for if we want to test the widgets

## Import Python packages
# try:
if "kso_utils" not in sys.modules:
    sys.path.append("..")
    import kso_utils.kso_utils

    sys.modules["kso_utils"] = kso_utils.kso_utils
# except:

# print("Installing latest version from PyPI...")
#%pip install -q kso-utils


# Import required modules
import kso_utils.tutorials_utils as t_utils
import kso_utils.project_utils as p_utils
import kso_utils.yolo_utils as y_utils
import kso_utils.widgets as kso_widgets
from kso_utils.project import ProjectProcessor, MLProjectProcessor
import kso_utils.server_utils as s_utils


# -----------------Initiating of the project structure -------------------------
# Initiate project's database
project = p_utils.find_project(project_name="Template project")
pp = ProjectProcessor(project)
mlp = MLProjectProcessor(pp, test=True)
# Create a folder for temporary test output
os.makedirs("../test/test_output", exist_ok=True)

# ----------------Tutorial 1----------------------------------------------------
"""
Does not test if the changes that can be made manually to the df are applied
Does not test if we can preview the movies
Both widgets can't be run due to a runtime error when nothing gets selected.
"""


def test_t1():
    # Display the map Tutorial 1
    pp.map_sites()
    # Retrieve and display movies
    pp.get_movie_info()
    pp.preview_media(test=True)

    # Check species dataframe
    pp.check_species_meta()

    # --- Manually updata metadata (same for sites, movies and species)
    sites_df, sites_range_rows, sites_range_columns = pp.select_meta_range(
        meta_key="sites"
    )

    sites_df_filtered, sites_sheet = kso_widgets.open_csv(
        df=sites_df,
        df_range_rows=sites_range_rows.value,
        df_range_columns=sites_range_columns.value,
    )

    sites_sheet_df = pp.view_meta_changes(
        df_filtered=sites_df_filtered, sheet=sites_sheet
    )

    pp.update_meta(sites_sheet_df, "sites", test=True)

    # --- Automatic check of movies metadata
    review_method = kso_widgets.choose_movie_review()
    gpu_available = kso_widgets.gpu_select()
    pp.check_movies_meta(
        review_method=review_method.value, gpu_available=gpu_available.result
    )
    assert len(sites_sheet_df) == 5


# #-------------Tutorial 2-------------------------------------------------------
# """
# This tutorial will be removed
# """

# #-------------Tutorial 3-------------------------------------------------------
"""
...
"""


def test_t3(zoo_user, zoo_pass):
    # Connect to zooniverse w/ Github credential
    pp.get_zoo_info(zoo_cred=[zoo_user, zoo_pass])
    pp.movie_selected = "movie_1.mp4"
    pp.check_movies_uploaded(pp.movie_selected)
    gpu_available = kso_widgets.gpu_select()
    pp.generate_zu_clips(
        movie_name=pp.movie_selected,
        movie_path=pp.movie_path,
        is_example=True,
        use_gpu=gpu_available.result,
        test=True,
    )

    assert len(pp.generated_clips) == 1


# #-------------Tutorial 4-------------------------------------------------------
# """
# It will not be tested if the actually uploading to Zooniverse works,
# since we do not want to upload things to them all the time.
# """
def test_t4(zoo_user, zoo_pass):
    pp.get_zoo_info(zoo_cred=[zoo_user, zoo_pass])
    pp.get_frames(test=True)
    pp.generate_zu_frames(test=True)
    # input_folder = kso_widgets.choose_folder()
    # output_folder = kso_widgets.choose_folder()
    # Generate suitable frames for upload by modifying initial frames
    # pp.generate_custom_frames(
    #    input_path=input_folder.selected,
    #    output_path=output_folder.selected,
    #    skip_start=0,
    #    skip_end=0,
    #    num_frames=10,
    #    frames_skip=None,
    # )
    t_utils.check_frame_size(
        frame_paths=pp.generated_frames["modif_frame_path"].unique()
    )
    t_utils.compare_frames(df=pp.generated_frames)
    assert len(pp.generated_frames) == 9


# #-------------Tutorial 5-------------------------------------------------------
# """
# ...
# """
def test_t5():
    # Generate current timestamp
    from datetime import datetime

    dt = datetime.now()

    # Download sample training data
    s_utils.get_ml_data(project, test=True)

    # Setup paths
    mlp.output_path = "../test/test_output"
    mlp.setup_paths(test=True)
    exp_name = f"notebook_test_train_{dt}"
    project_path = os.path.join(mlp.output_path, mlp.project_name)

    # Model training
    weights = y_utils.choose_baseline_model(mlp.output_path, test=True)
    batch_size, epochs, img_h, img_w = mlp.choose_train_params()
    mlp.modules["wandb"].finish()
    mlp.train_yolov5(
        exp_name=exp_name,
        weights=weights.artifact_path,
        project=project_path,
        epochs=epochs.value,
        batch_size=batch_size.value,
        img_size=img_h.value,
    )

    assert len(os.listdir(os.path.join(project_path, exp_name, "weights"))) == 2

    # Model evaluation
    conf_thres = t_utils.choose_eval_params()
    # Evaluate YOLO Model on Unseen Test data
    mlp.eval_yolov5(exp_name, conf_thres.value)

    # Enhancement tests (leave out for now)
    # eh_conf_thres = t_utils.choose_eval_params()
    # mlp.enhance_yolov5(conf_thres=eh_conf_thres.value,
    #                   in_path=os.path.join(mlp.output_path, "ml-template-data"),
    #                   project_path=project_path, img_size=[640, 640])
    # mlp.enhance_replace(os.path.join(mlp.output_path, "ml-template-data"))

    # assert len(os.listdir(f"{mlp.run_path}/labels")==0)


# #-------------Tutorial 6-------------------------------------------------------
# """
# ...
# """
# def test_t6():
# # Evaluation
# model = mlp.choose_model()
# download_dir = kso_widgets.choose_folder(".", "where to download the model")
# artifact_dir = mlp.get_model(model.value, download_dir.selected)
# source = kso_widgets.choose_footage(
#     project=pp.project,
#     server_connection=pp.server_connection,
#     db_connection=pp.db_connection,
#     start_path=pp.project.movie_folder
#     if pp.project.movie_folder not in [None, "None"]
#     else ".",
#     folder_type="custom footage",
# )
# source_value = t_utils.process_source(source)
# save_dir = kso_widgets.choose_folder(".", "runs output")
# conf_thres = t_utils.choose_conf()
# mlp.detect_yolov5(
#     source=source_value,
#     save_dir=save_dir.selected,
#     conf_thres=conf_thres.value,
#     artifact_dir=artifact_dir,
# )
# eval_dir = kso_widgets.choose_folder(
#     save_dir.selected
#     if "save_dir" in vars() and save_dir.selected is not None
#     else ".",
#     "runs output",
# )
# mlp.save_detections_wandb(conf_thres.value, model.value, eval_dir.selected)
# viewer = y_utils.choose_files(eval_dir.selected)
# train_dataset, val_dataset = mlp.get_dataset(model.value)
# y_utils.get_data_viewer(os.path.join(train_dataset, "data/images"))
# y_utils.get_data_viewer(os.path.join(val_dataset, "data/images"))

# # Tracking individuals
# mlp.track_individuals(
#     source=source_value,
#     artifact_dir=artifact_dir,
#     eval_dir=eval_dir.selected,
#     conf_thres=conf_thres.value,
#     img_size=(540, 540),
# )


# #-------------Tutorial 7-------------------------------------------------------
# """
# ...
# """

# def test_t7():
# model = mlp.choose_model()
# download_dir = kso_widgets.choose_folder(".", "downloaded model")
# artifact_dir = mlp.get_model(model.value, download_dir.selected)
# ACCESS_TOKEN = ""
# archive_dir = kso_widgets.choose_folder(".", "archive for upload")
# depo_id = zenodo_utils.upload_archive(ACCESS_TOKEN, artifact_dir=artifact_dir)
# upload_title = t_utils.choose_text("title")
# upload_description = t_utils.choose_text("description")
# authors = t_utils.WidgetMaker()
# zenodo_utils.add_metadata_zenodo_upload(
#     ACCESS_TOKEN,
#     depo_id,
#     upload_title.value,
#     upload_description.value,
#     authors.author_dict,
# )


# #-------------Tutorial 8-------------------------------------------------------
# """
# ...
# """
# pp.choose_workflows(False, zoo_cred=zoo_cred)

# workflow_checks = {'Workflow name: #0': 'Development workflow',
#  'Subject type: #0': 'clip',
#  'Minimum workflow version: #0': 1.0}

# class_df = pp.get_classifications(
#     workflow_checks,
#     pp.zoo_info["workflows"],
#     workflow_checks["Subject type: #0"],
#     pp.zoo_info["classifications"],
# )
