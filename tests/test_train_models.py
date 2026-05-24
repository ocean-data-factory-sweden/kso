from pathlib import Path
import shutil
from kso_utils.train_models import train_models


def test_train_models(needs_wandb):
    """
    When the training of yolo (ultralytics) is called via pytest, the automatic logging in WandB does not work.
    See: https://github.com/ultralytics/ultralytics/blob/8ccb5fb8dd1b2e937314e56d041c592c91085d30/ultralytics/utils/callbacks/wb.py#L8

    So instead of the model results being stored at wand/run_id/..., when we run train_models via pytest, they get created at
    ./template_project/test_template.
    This test runs train_models to see if it runs. It also checks if the correct files are created.
    The WandB logging is not tested, this is partly done in the test of the registry.
    """
    mlp = train_models(
        project_name="Template project",
        data_path="ml-template-data",
        exp_name="test_template",
        model_name="yolov8m-base-model",
        model_download_dir=".",
        batch_size=1,
        epochs=1,
        img_h=128,
    )

    output_dir = Path(mlp.last_train_dir)

    # Test if the folder is created and contains the files is should have after training
    assert (
        output_dir.is_dir()
    ), f"Successful training should create output dir '{output_dir}', but it is missing."
    assert output_dir.joinpath(
        "weights"
    ).is_dir(), f"Successful training should create '{output_dir / 'weights'}', but it is missing."
    assert output_dir.joinpath(
        "weights", "best.pt"
    ).is_file(), (
        f"Missing trained model weights at '{output_dir / 'weights' / 'best.pt'}'."
    )
    assert output_dir.joinpath(
        "results.csv"
    ).is_file(), f"Missing training metrics file at '{output_dir / 'results.csv'}'."

    # Remove the training results so that we start with a clean slate next time.
    shutil.rmtree(output_dir)
