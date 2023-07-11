# -*- coding: utf-8 -*-
"""

This auto-test does not test if everything displays what it should display. 
It mainly tests if everything still runs without giving any errors.

This test contains the test for all notebooks. 

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
"""
Does not test if the changes that can be made manually to the df are applied
Does not test if we can preview the movies
Both widgets can't be run due to a runtime error when nothing gets selected.
"""


def test_t1():
    import kso_utils.widgets as kso_widgets

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


# -------------Tutorial 4-------------------------------------------------------
# """
# It will not be tested if the actually uploading to Zooniverse works,
# since we do not want to upload things to them all the time.
# """
def test_t4(zoo_user, zoo_pass):

    import kso_utils.tutorials_utils as t_utils

    # import kso_utils.widgets as kso_widgets

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
    pp.get_zoo_info(zoo_cred=[zoo_user, zoo_pass])
    pp.choose_workflows(generate_export=False, zoo_cred=[zoo_user, zoo_pass])

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

    assert len(class_df) == 9

    agg_params = kso_widgets.choose_agg_parameters(workflow_checks["Subject type: #0"])

    agg_df, raw_df = pp.process_classifications(
        pp.zoo_info["classifications"],
        workflow_checks["Subject type: #0"],
        agg_params,
        summary=False,
    )

    assert len(agg_df) == 3

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
