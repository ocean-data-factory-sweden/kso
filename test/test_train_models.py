from pathlib import Path
import shutil
from kso_utils.train_models import train_models


def test_train_models():
    """
    When the training of yolo (ultralytics) is called via pytest, the automatic logging in WandB does not work.
    See: https://github.com/ultralytics/ultralytics/blob/8ccb5fb8dd1b2e937314e56d041c592c91085d30/ultralytics/utils/callbacks/wb.py#L8

    So instead of the model results being stored at wand/run_id/..., when we run train_models via pytest, they get created at
    ./template_project/test_template.
    This test runs train_models to see if it runs. It also checks if the correct files are created.
    The WandB logging is not tested, this is partly done in the test of the registry.
    """
    # Make sure that the results folder does not exist yet before this test
    assert not Path(
        "template_project/test_template"
    ).is_dir(), "The folder 'template_project/test_template' already exits. This must not exist at the beginning of the test. Delete it."

    _ = train_models(
        project_name="Template project",
        data_path="ml-template-data",
        exp_name="test_template",
        model_name="yolov8m-base-model",
        model_download_dir=".",
        batch_size=1,
        epochs=1,
        img_h=128,
    )

    # Test if the folder is created and contains the files is should have after training
    assert Path(
        "template_project/test_template"
    ).is_dir(), "Successfull training creates the folder 'template_project/test_template'. This is missing, so training failed."
    assert Path(
        "template_project/test_template/weights"
    ).is_dir(), "Successfull training saves the models in the folder 'template_project/test_template/weight'. This is missing, so training failed."
    assert Path(
        "template_project/test_template/weights/best.pt"
    ).is_file(), "The file 'template_project/test_template/weight/best.pt' is missing. Something must have gone wrong."
    assert Path(
        "template_project/test_template/results.csv"
    ).is_file(), "The file 'template_project/test_template/results.csv' is missing. Something must have gone wrong."

    # Remove the training results so that we start with a clean slate next time.
    shutil.rmtree("template_project/test_template")
