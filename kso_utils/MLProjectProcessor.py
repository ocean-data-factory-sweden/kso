# base imports
import time
import sys
import logging
import pandas as pd
import ipywidgets as widgets
import shutil
from pathlib import Path
from IPython.display import display
import torch
import yaml
import ultralytics
import pims
import cv2
from importlib import import_module

# util imports
import kso_utils.project_utils as project_utils
import kso_utils.widgets as kso_widgets
import kso_utils.yolo_utils as yolo_utils
import kso_utils.zenodo_utils as zenodo_utils
from kso_utils.ProjectProcessor import ProjectProcessor
import kso_utils.general as g_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class MLProjectProcessor(ProjectProcessor):
    def __init__(
        self,
        project: project_utils.Project,
        config_path: str = None,
        weights_path: str = None,
        output_path: str = None,
        classes: list = [],
    ):
        super().__init__(project)
        self.project_name = self.project.Project_name.lower().replace(" ", "_")
        self.data_path = config_path
        self.weights_path = weights_path
        self.output_path = output_path
        self.classes = classes
        self.run_history = None
        self.best_model_path = None
        self.model_type = 1  # set as 1 for testing
        self.train, self.run, self.test = (None,) * 3
        self.registry = "wandb"  # TODO: make this a config and read it in here instead of hardcoding
        # TODO: To be able to support more registries, new registries must be written, and
        # a configuraiton option added to the project config. In the meanwhile do the import 
        # dynamically, even with a hard-coded wandb, so that adding more registries requires
        # no more refactoring
        self.registry_utils = import_module(f"kso_utils.registries.wandb_utils")

        g_utils.validate_utils(
            self.registry_utils,
            [
                "init",
                "start_run",
                "close_run",
                "choose_baseline_model",
                "choose_model",
                "get_model",
                "get_dataset",
            ],
        )

        # Before t6_utils gets loaded in, the val.py file in yolov5_tracker repository needs to be removed
        # to prevent the batch_size error, see issue kso-object-detection #187
        path_to_val = Path(sys.path[0], "yolov5_tracker/val.py")
        try:
            if path_to_val.exists():
                path_to_val.unlink()
        except OSError:
            pass

        # Monkey-patch the cv2.VideoWriter class to use the default codec
        class CustomVideoWriter(cv2.VideoWriter):
            def __init__(self, *args, **kwargs):
                args = list(args)
                if len(args) > 0:
                    args[0] = args[0].replace(".avi", ".mp4")
                    args[1] = cv2.VideoWriter_fourcc(*"avc1")
                super().__init__(*args, **kwargs)

        # Replace cv2.VideoWriter with the patched version
        cv2.VideoWriter = CustomVideoWriter

        self.team_name = "koster"  # TODO: Should be part of the config
        logging.info("ML Project successfully initialised.")

    def prepare_dataset(
        self,
        agg_df: pd.DataFrame,
        out_path: str,
        perc_test: float = 0.2,
        img_size: tuple = (224, 224),
        remove_nulls: bool = False,
        track_frames: bool = False,
        n_tracked_frames: int = 0,
        out_format: str = "yolo",
    ):
        species_list = kso_widgets.choose_species(
            self.db_connection, agg_df.label.unique().tolist()
        )

        button = widgets.Button(
            description="Aggregate frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            # code for prepare dataset for machine learning
            yolo_utils.frame_aggregation(
                project=self.project,
                server_connection=self.server_connection,
                db_connection=self.db_connection,
                out_path=out_path,
                perc_test=perc_test,
                class_list=self.species_of_interest,
                img_size=img_size,
                remove_nulls=remove_nulls,
                track_frames=track_frames,
                n_tracked_frames=n_tracked_frames,
                agg_df=agg_df,
                out_format=out_format,
            )

            button.on_click(on_button_clicked)
            display(button)

    #############
    # t5
    #############
    def choose_baseline_model(self, download_path: str):
        """
        It downloads the latest version of the baseline model from the model registry
        :return: The path to the baseline model.
        """
        return self.registry_utils.choose_baseline_model(download_path)

    def choose_entity(self, alt_name: bool = False):
        if self.team_name is None:
            return kso_widgets.choose_entity()
        else:
            if not alt_name:
                logging.info(
                    f"Found team name: {self.team_name}. If you want"
                    " to use a different team name for this experiment"
                    " set the argument alt_name to True"
                )
            else:
                return kso_widgets.choose_entity()

    def setup_paths(self):
        if not isinstance(self.output_path, str) and self.output_path is not None:
            self.output_path = self.output_path.selected
        self.data_path, self.hyp_path = yolo_utils.setup_paths(
            self.output_path, self.model_type
        )

    def choose_train_params(self):
        return kso_widgets.choose_train_params(self.model_type)

    def train_yolo(
        self,
        exp_name: str,
        weights: str,
        project: str,
        epochs: int = 1,
        batch_size: int = 16,
        img_size: int = 128,
    ):
        try:
            if "yolov5" in weights:
                weights = str(Path(weights).name)

            model = ultralytics.YOLO(weights)
            model.train(
                data=self.data_path,
                project=project,
                name=exp_name,
                epochs=int(epochs),
                batch=int(batch_size),
                imgsz=img_size,
            )
        except Exception as e:
            logging.info(f"Training failed due to: {e}")
            self.registry_utils.close_run()

    def enhance_yolo(
        self, in_path: str, project_path: str, conf_thres: float, img_size=[640, 640]
    ):
        from datetime import datetime

        run_name = f"enhance_run_{datetime.now()}"
        self.run_path = Path(project_path, run_name)
        logging.info("Enhancement running...")
        model = ultralytics.YOLO(self.tuned_weights)
        model.predict(
            source=str(Path(in_path, "images")),
            conf=conf_thres,
            save_txt=True,
            save_conf=True,
            save=True,
            imgsz=img_size,
        )

    def enhance_replace(self, data_path: str):
        if self.model_type == 1:
            # Rename the 'labels' directory to 'labels_org'
            data_path = Path(data_path)
            data_path.joinpath("labels").rename(data_path.joinpath("labels_org"))
            # Rename the 'labels' directory inside 'self.run_path' to 'labels'
            self.run_path.joinpath("labels").rename(data_path.joinpath("labels"))
        else:
            logging.error("This option is not supported for other model types.")

    #############
    # t6
    #############
    # Function to choose a model to evaluate

    def choose_model(self, custom_project: str = "", publish: bool = False):
        """
        It takes a project name that is defined in the class and returns a dropdown widget that displays the metrics of the model
        selected

        :param project_name: The name of the project you want to load the model from
        :return: The model_widget is being returned.
        """
        model_dict = zenodo_utils.download_and_extract_models_from_zenodo(
            "pClzrdKwErArGWuPXMje0OtLEaq2gM8vHcAEeQN9CXyS2IjbuJsw05JLjVII"
        )
        if publish:
            # Create the dropdown widget
            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                description="Select Zenodo model: ",
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"description_width": "initial"},
            )
            # Display the dropdown widget
            display(model_widget)
            return model_widget
        return self.registry_utils.choose_model(self, model_dict, custom_project)

    def eval_yolo(self, exp_name: str, conf_thres: float):
        # Find trained model weights
        project_path = Path(self.project_name, exp_name)
        self.tuned_weights = f"{Path(project_path, 'weights', 'best.pt')}"
        try:
            model = ultralytics.YOLO(self.tuned_weights)
            model.val(
                data=self.data_path,
                conf=conf_thres,
            )
            logging.info("Run succeeded, finishing run...")
        except Exception as e:
            logging.error(f"Encountered {e}, terminating run...")
            self.registry_utils.close_run()

    def _process_results(self, src, results):
        fc = 0
        if Path(src).is_dir():
            obj = [f for f in Path(src).iterdir() if f.is_file()]
        else:
            obj = pims.Video(src)  # store video capture object
        inc = 0
        for r in results:
            fc += 1
            if fc > 1:
                end = time.time()
                inc = end - st
            t = sum(r.speed.values()) / 1000
            t_left = (len(obj) - fc) * max(t, inc)  # conservative
            st = time.time()
            statement = f"Processed item {fc} / {len(obj)} in {t*1000} ms. Estimated remaining time: {round(t_left, 2)}s."
            if t_left < 60:
                logging.info(f"{statement} Almost there! ⏳")
            else:
                logging.info(f"{statement} Grab a ☕")
        logging.info("Prediction completed successfully! ✅")

    def detect_yolo(
        self,
        project: str,
        name: str,
        source: str,
        save_dir: str,
        conf_thres: float,
        artifact_dir: str,
        model: str,
        img_size: int = 640,
        save_output: bool = True,
        latest: bool = True,
        out_format: str = "yolo",
    ):
        from yolov5.utils.general import increment_path

        self.registry_utils.start_run(self, "model-evaluations", None)
        models = [
            str(f)
            for f in Path(artifact_dir).iterdir()
            if f.is_file()
            and str(f).endswith((".pt", ".model"))
            and "osnet" not in str(f)
            and "best" in str(f)
        ]
        if len(models) > 0:
            best_model = models[0]
        else:
            logging.info("No trained model found, using yolov8 base model...")
            best_model = "yolov8s.pt"

        model = ultralytics.YOLO(best_model)
        project = str(Path(save_dir))
        self.eval_dir = str(increment_path(Path(project) / name, exist_ok=False))
        if latest:
            if isinstance(source, list):
                for src in source:
                    results = model.predict(
                        project=project,
                        name=name,
                        source=src,
                        conf=conf_thres,
                        save_txt=True,
                        save_conf=True,
                        save=save_output,
                        imgsz=img_size,
                        stream=True,
                        verbose=False,
                    )
                    self._process_results(src, results)

            else:
                results = model.predict(
                    project=project,
                    name=name,
                    source=source,
                    conf=conf_thres,
                    save_txt=True,
                    save_conf=True,
                    save=save_output,
                    imgsz=img_size,
                    stream=True,
                    verbose=False,
                )
                self._process_results(source, results)
        else:
            logging.error(
                "We do not currently support running YoloV5 models. Please re-train models "
                "using the latest model version available"
            )
            return
        self._save_detections(conf_thres, model.ckpt_path, self.eval_dir, out_format)
        self.registry_utils.close_run()

    def _save_detections(
        self, conf_thres: float, model: str, eval_dir: str, out_format: str = "yolo"
    ):
        if self.registry == "wandb":

            def read_yaml_file(file_path):
                with open(file_path, "r") as file:
                    yaml_data = yaml.safe_load(file)
                return yaml_data

            # Read species mapping into data dictionary
            try:
                data_dict = read_yaml_file(self.data_path)
                species_mapping = data_dict["names"]
            except FileNotFoundError:
                # Handle the case when the file doesn't exist
                logging.info(f"File not found: {self.data_path}")
                species_mapping = {}
            except KeyError:
                # Handle the case when the "names" key is missing
                logging.info("Key 'names' not found in the YAML file.")
                species_mapping = {}
            except Exception as e:
                # Handle any other unexpected errors
                logging.info(f"An unexpected error occurred: {e}")
                species_mapping = {}

            yolo_utils.set_config(
                conf=conf_thres,
                model_name=model,
                evaluation_directory=eval_dir,
                species_map=species_mapping,
            )
            self.csv_report = yolo_utils.generate_csv_report(
                evaluation_path=eval_dir,
                log=True,
                registry=self.registry,
                movie_csv_df=self.local_movies_csv,
                out_format=out_format,
            )
            yolo_utils.add_data(
                Path(eval_dir, "annotations.csv"),
                "detection_output",
                self.registry,
                self.run,
            )

            shutil.make_archive(
                Path(eval_dir, "labels"), "zip", Path(eval_dir, "labels")
            )
            yolo_utils.add_data(
                Path(eval_dir, "labels"),
                "detection_output",
                self.registry,
                self.run,
            )
        elif self.registry is None:
            self.csv_report = yolo_utils.generate_csv_report(
                evaluation_path=eval_dir,
                log=True,
                registry=self.registry,
                movie_csv_df=self.local_movies_csv,
                out_format=out_format,
            )
        else:
            logging.error("Invalid registry name")
            return

    def _increment_path(self, path, exist_ok=False, sep="", mkdir=False):
        # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            path, suffix = (
                (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
            )

            # Method 1
            for n in range(2, 9999):
                p = path.with_name(f"{path.stem}{sep}{n}{suffix}")  # increment path
                if not p.exists():
                    break

            # Method 2 (deprecated)
            # dirs = glob.glob(f"{path}{sep}*")  # similar paths
            # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
            # i = [int(m.groups()[0]) for m in matches if m]  # indices
            # n = max(i) + 1 if i else 2  # increment number
            # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

        if mkdir:
            p.mkdir(parents=True, exist_ok=True)  # make directory

        return p

    def track_individuals(
        self,
        name: str,
        source: str,
        artifact_dir: str,
        conf_thres: float,
        img_size: tuple = (540, 540),
    ):
        if not hasattr(self, "eval_dir"):
            self.eval_dir = str(
                self._increment_path(
                    path=Path(self.save_dir) / "detect", exist_ok=False
                )
            )

        latest_tracker = yolo_utils.track_objects(
            name=name,
            source_dir=source,
            artifact_dir=artifact_dir,
            tracker_folder=self.eval_dir,
            conf_thres=conf_thres,
            img_size=img_size,
            gpu=True if torch.cuda.is_available() else False,
        )

        # Create a new run for tracking only if necessary
        self.registry_utils.start_run(self, "model-evaluations", "track")

        if self.registry == "wandb":
            yolo_utils.set_config(
                conf=conf_thres,
                model_name=artifact_dir,
                evaluation_directory=self.eval_dir,
            )

        # self.csv_report = yolo_utils.generate_csv_report(
        #    self.team_name, self.project_name, eval_dir, self.run, log=True
        # )
        self.tracking_report = yolo_utils.generate_counts(
            self.eval_dir,
            latest_tracker,
            artifact_dir,
            self.run,
            log=True,
            registry=self.registry,
        )
        yolo_utils.add_data(
            str(Path(latest_tracker).parent.absolute()),
            "tracker_output",
            self.registry,
            self.run,
        )
        self.registry_utils.close_run()

    def get_model(self, model_name: str, download_path: str, custom_project: str = ""):
        """
        It downloads the latest model checkpoint from the specified project and model name

        :param model_name: The name of the model you want to download
        :type model_name: str
        :param project_name: The name of the project you want to download the model from
        :type project_name: str
        :param download_path: The path to download the model to
        :type download_path: str
        :return: The path to the downloaded model checkpoint.
        """
        if ".pt" in model_name and ":" not in model_name:
            logging.info("Local model successfully loaded.")
            return str(Path(model_name).parent)

        return self.registry_utils.get_model(
            self, model_name, download_path, custom_project
        )

    def get_dataset(
        self,
        model: str,
        team_name: str = "koster",
    ):
        """
        It takes in a project name and a model name, and returns the paths to the train and val datasets

        :param project_name: The name of the project you want to download the dataset from
        :type project_name: str
        :param model: The model you want to use
        :type model: str
        :return: The return value is a list of two directories, one for the training data and one for the validation data.
        """
        return self.registry_utils.get_dataset(self, model, team_name)
