# -*- coding: utf-8 -*-
"""

This auto-test does not test if everything displays what it should display. 
It mainly tests if everything still runs without giving any errors.

This collection of tests covers the following tutorial notebooks:
 - 1, 3, 4, 5, 6, 8

All other tutorials are not tested automatically and care should be taken when
making changes as these could break existing workflows. 

To run these tests manually, use pytest --disable-warnings test/notebook-tests.py

"""

# --------------------The first 2 cells of every notebook-----------------------
## Import Python packages
import os
import sys

if "kso_utils" not in sys.modules:
    # for when we are running it on git, the yaml file will install the docker image. then we do need to go to the correct directory
    os.chdir("tutorials")
    sys.path.append("..")
    import kso_utils.kso_utils

    sys.modules["kso_utils"] = kso_utils.kso_utils

import kso_utils.project_utils as p_utils
from kso_utils.project import ProjectProcessor, MLProjectProcessor

# https://stackoverflow.com/questions/15044447/how-do-i-unit-testing-my-gui-program-with-python-and-pyqt
# for if we want to test the widgets

# -----------------Initiating of the project structure -------------------------
# Initiate project's database
project = p_utils.find_project(project_name="Template project")
pp = ProjectProcessor(project)
mlp = MLProjectProcessor(pp, test=True)
# Create a folder for temporary test output
os.makedirs("../test/test_output", exist_ok=True)

# ----------------Tutorial 1----------------------------------------------------


def test_t1():
    import kso_utils.widgets as kso_widgets

    # Display the map Tutorial 1
    pp.map_sites()
    # Retrieve and display movies
    pp.get_movie_info()
    # Preview the media (pre-selected)
    pp.preview_media(test=True)

    # Check species dataframe
    pp.check_species_meta()

    # Manually updata metadata (same for sites, movies and species)
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
    # Test for length of final sites sheet dataframe. Since each meta table uses
    # these functions interchangeably, we only test for sites metadata.
    assert len(sites_sheet_df) == 25


# -------------Tutorial 2-------------------------------------------------------
# """
# This tutorial will be removed
# """

# -------------Tutorial 3-------------------------------------------------------
"""
...
"""


def test_t3(zoo_user, zoo_pass):
    import kso_utils.widgets as kso_widgets

    # Connect to zooniverse w/ Github credential
    pp.connect_zoo_project(zoo_cred=[zoo_user, zoo_pass])
    # Pre-selected test movie
    pp.movie_selected = "movie_1.mp4"
    # Check whether movie has been uploaded previously
    pp.check_movies_uploaded(pp.movie_selected)
    # Do not use GPU by default
    gpu_available = kso_widgets.gpu_select()
    # Generate a default number of clips for testing
    pp.generate_zoo_clips(
        movie_name=pp.movie_selected,
        movie_path=pp.movie_path,
        is_example=True,
        use_gpu=gpu_available.result,
        test=True,
    )
    # Ensure that the clips are returned as part of generated_clips
    assert len(pp.generated_clips) == 1


# -------------Tutorial 4-------------------------------------------------------
# """
# It will not be tested if the actually uploading to Zooniverse works,
# since we do not want to upload things to them all the time.
# """
def test_t4(zoo_user, zoo_pass):

    import kso_utils.tutorials_utils as t_utils

    # import kso_utils.widgets as kso_widgets

    # Log into Zooniverse and load relevant data from their DB
    pp.connect_zoo_project(zoo_cred=[zoo_user, zoo_pass])
    # Process zoo classifications
    pp.process_zoo_classifications(test=True)
    # Aggregate classifications
    pp.aggregate_zoo_classifications(test=True, agg_params=[0.1, 1])
    # Fetch relevant frame subjects in dataframe (by default all species for testing)
    pp.extract_zoo_frames(test=True)
    # Generate frames from based on subject metadata
    pp.modify_zoo_frames(test=True)
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
    # Ensure that extracted frames are of suitable size for upload to ZU
    t_utils.check_frame_size(
        frame_paths=pp.generated_frames["modif_frame_path"].unique()
    )
    # Compare original vs modified frames (no interactivity tested)
    t_utils.compare_frames(df=pp.generated_frames)
    # Test that final generated frames contain 9 rows (representing a classification)
    assert len(pp.generated_frames) == 9


# # #-------------Tutorial 5-------------------------------------------------------
# # """
# # ...
# # """
def test_t5():

    import kso_utils.tutorials_utils as t_utils
    import kso_utils.yolo_utils as y_utils
    import kso_utils.server_utils as s_utils

    # Generate current timestamp
    import shutil
    from datetime import datetime

    dt = datetime.now()

    # Download sample training data
    s_utils.get_ml_data(project, test=True)

    # Setup paths
    mlp.output_path = "../test/test_output"
    mlp.setup_paths(test=True)
    exp_name = f"notebook_test_train_{dt}"
    project_path = os.path.join(mlp.output_path, mlp.project_name)
    exp_path = os.path.join(project_path, exp_name)

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
    # Test whether last.pt and best.pt are present in the weights folder (i.e. model training
    # was successful)
    assert len(os.listdir(os.path.join(exp_path, "weights"))) == 2

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

    # Clean up residual files
    shutil.rmtree(exp_path)


# -------------Tutorial 6-------------------------------------------------------
"""
...
"""


def test_t6():
    import kso_utils.tutorials_utils as t_utils
    import kso_utils.server_utils as s_utils

    # Generate current timestamp
    import shutil
    from datetime import datetime

    dt = datetime.now()

    # Create a unique experiment name
    exp_name = f"custom_{dt}".replace(" ", "_").replace(".", "_").replace(":", "-")
    mlp.output_path = "../test/test_output"
    project_path = os.path.join(mlp.output_path, mlp.project_name)
    exp_path = os.path.join(project_path, exp_name)

    # Evaluation
    s_utils.get_ml_data(project, test=True)
    model = mlp.choose_model().options[-1][1]
    download_dir = mlp.output_path
    artifact_dir = mlp.get_model(model, download_dir)
    source = os.path.join("../test/test_output", mlp.project.ml_folder, "images")
    save_dir = project_path
    conf_thres = t_utils.choose_conf()
    mlp.detect_yolov5(
        exp_name=exp_name,
        source=source,
        save_dir=save_dir,
        conf_thres=conf_thres.value,
        artifact_dir=artifact_dir,
    )
    eval_dir = exp_path
    mlp.save_detections_wandb(conf_thres.value, model, eval_dir)

    # Note: investigating training and validation datasets is not currently tested.

    # Create unique tracking experiment name
    track_exp_name = (
        f"tracker_test_{dt}".replace(" ", "_").replace(".", "_").replace(":", "-")
    )

    # Tracking individuals
    mlp.track_individuals(
        name=track_exp_name,
        source=source,
        artifact_dir=artifact_dir,
        eval_dir=eval_dir,
        conf_thres=conf_thres.value,
        img_size=(640, 640),
    )
    # Remove any residual files
    shutil.rmtree(exp_path)


# #-------------Tutorial 7-------------------------------------------------------
"""
This tutorial is still in a very early stage and mostly uses widgets directly from widgets.py and
API functionality from Zenodo. Tests should be added in future as its capabilities expand. 
"""


def test_t7():
    pass


# -------------Tutorial 8-------------------------------------------------------
"""
...
"""


def test_t8(zoo_user, zoo_pass):
    import kso_utils.widgets as kso_widgets

    # Connect to zooniverse w/ Github credential
    pp.connect_zoo_project(zoo_cred=[zoo_user, zoo_pass])
    # pp.choose_workflows(generate_export=False, zoo_cred=[zoo_user, zoo_pass])

    workflow_checks = {
        "Workflow name: #0": "Development workflow",
        "Subject type: #0": "clip",
        "Minimum workflow version: #0": 1.0,
    }

    class_df = pp.get_classifications(
        workflow_checks,
        pp.zoo_info["workflows"],
        workflow_checks["Subject type: #0"],
        pp.zoo_info["classifications"],
    )

    # Check that all classifications are retrieved
    # from workflows
    assert len(class_df) == 9

    import kso_utils.zooniverse_utils as zoo_utils

    zoo_utils.process_zoo_classifications(
        project=pp.project,
        db_connection=pp.db_connection,
        csv_paths=pp.csv_paths,
        classifications_data=pp.zoo_info["classifications"],
        subject_type=workflow_checks["Subject type: #0"],
    )

    agg_params = kso_widgets.choose_agg_parameters(workflow_checks["Subject type: #0"])

    pp.aggregate_zoo_classifications(agg_params, test=True)

    # Check that aggregation was successful
    assert len(pp.aggregated_zoo_classifications) == 3

    # Run the preparation script
    mlp.output_path = "../test/test_output"
    # This test does not work yet as the Template Project has no frame classifications
    # TODO: Add frame classifications and adjust workflow name and subject type above
    # mlp.prepare_dataset(
    #    agg_df=agg_df,
    #    out_path=mlp.output_path,
    #    img_size=(640, 640),
    #    perc_test=0.2,
    #    test=True,
    # )
