# base imports
import os
import time
import sys
import logging
import pandas as pd
import ipywidgets as widgets
import shutil
from pathlib import Path
from IPython.display import display, clear_output
import torch
import wandb
import yaml
import ultralytics
import pims
import itertools
import cv2

# util imports
import kso_utils.project_utils as project_utils
import kso_utils.widgets as kso_widgets
import kso_utils.yolo_utils as yolo_utils
import kso_utils.zenodo_utils as zenodo_utils
from kso_utils.ProjectProcessor import ProjectProcessor

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

        self.team_name = "koster"
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
        It downloads the latest version of the baseline model from WANDB
        :return: The path to the baseline model.
        """
        if self.registry == "wandb":
            api = wandb.Api()
            # weird error fix (initialize api another time)
            api.runs(path="koster/model-registry")
            api = wandb.Api()
            collections = [
                coll
                for coll in api.artifact_type(
                    type_name="model", project="koster/model-registry"
                ).collections()
            ]

            model_dict = {}
            for artifact in collections:
                model_dict[artifact.name] = artifact

            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                value=None,
                description="Select model:",
                ensure_option=False,
                disabled=False,
                layout=widgets.Layout(width="50%"),
                style={"description_width": "initial"},
            )

            main_out = widgets.Output()
            display(model_widget, main_out)

            def on_change(change):
                with main_out:
                    clear_output()
                    try:
                        for af in model_dict[change["new"].name]:
                            artifact_dir = af.download(download_path)
                            artifact_file = [
                                str(i)
                                for i in Path(artifact_dir).iterdir()
                                if str(i).endswith(".pt")
                            ][-1]
                            logging.info(
                                f"Baseline {af.name} successfully downloaded from WANDB"
                            )
                            model_widget.artifact_path = artifact_file
                    except Exception as e:
                        logging.error(
                            f"Failed to download the baseline model. Please ensure you are logged in to WANDB. {e}"
                        )
                        model_widget.artifact_path = "yolov8m.pt"

            model_widget.observe(on_change, names="value")
            return model_widget
        else:
            logging.error("Registry not supported.")

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
        # Disable wandb (not necessary yet)
        if self.registry == "wandb":
            ultralytics.settings.update({"wandb": True})

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
        # Close down run
        if self.registry == "wandb":
            wandb.finish()

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

        if wandb.run is not None:
            wandb.finish()

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
        # TODO: Remove hardcoded API key from Zenodo
        model_dict = zenodo_utils.download_and_extract_models_from_zenodo(
            "pClzrdKwErArGWuPXMje0OtLEaq2gM8vHcAEeQN9CXyS2IjbuJsw05JLjVII"
        )
        model_info = {v: {"data": "No model info"} for k, v in model_dict.items()}
        data_info = {v: {"data": "No data info"} for k, v in model_dict.items()}
        if self.registry == "wandb" and not publish:
            api = wandb.Api()

            # weird error fix (initialize api another time)
            if len(custom_project) > 0:
                logging.info(
                    "Please note: Using models from custom project, please ensure that you have access."
                )
                full_path = custom_project
                api.runs(path=full_path).objects
            elif self.project_name == "template_project":
                full_path = f"{self.team_name}/spyfish_aotearoa"

            else:
                full_path = f"{self.team_name}/{self.project_name}"

            runs = api.runs(full_path)

            if len(runs) > 100:
                runs = list(runs)[:100]

            for run in runs:
                model_artifacts = [
                    artifact
                    for artifact in itertools.chain(
                        run.logged_artifacts(), run.used_artifacts()
                    )
                    if artifact.type == "model"
                ]
                if len(model_artifacts) > 0:
                    model_dict[run.name] = model_artifacts[0].name.split(":")[0]
                    model_info[model_artifacts[0].name.split(":")[0]] = run.summary
                    data_info[model_artifacts[0].name.split(":")[0]] = run.config

            # Add "no movie" option to prevent conflicts
            # models = np.append(list(model_dict.keys()),"No model")

            model_widget = widgets.Dropdown(
                options=[(name, model) for name, model in model_dict.items()],
                description="Select model:",
                ensure_option=False,
                disabled=False,
                value=None,
                layout=widgets.Layout(width="50%"),
                style={"description_width": "initial"},
            )

            main_out = widgets.Output()
            display(model_widget, main_out)

            # Display model metrics
            def on_change(change):
                with main_out:
                    clear_output()
                    if change["new"] == "No file":
                        logging.info("Choose another file")
                    else:
                        if self.project_name == "model-registry":
                            logging.info("No metrics available")
                        else:
                            self.data_path = data_info[change["new"]]["data"]
                            logging.info(
                                {
                                    k: v
                                    for k, v in model_info[change["new"]].items()
                                    if "metrics" in k
                                }
                            )

            model_widget.observe(on_change, names="value")
            return model_widget

        elif self.registry is None or publish:
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
        else:
            logging.error("The chosen registry is not available at the moment.")
            return

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
        except Exception as e:
            logging.error(f"Encountered {e}, terminating run...")
            wandb.finish()
        logging.info("Run succeeded, finishing run...")
        wandb.finish()

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

        if self.registry == "wandb":
            self.run = wandb.init(
                entity=self.team_name,
                project="model-evaluations",
                settings=wandb.Settings(start_method="thread"),
            )
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
        if self.registry == "wandb":
            self.run = wandb.init(
                entity=self.team_name,
                project="model-evaluations",
                name="track",
                settings=wandb.Settings(start_method="thread"),
            )
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
        if self.registry == "wandb":
            wandb.finish()

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

        if self.registry == "wandb":
            # weird error fix (initialize api another time)
            if len(custom_project) > 0:
                logging.info(
                    "Please note: Using models from custom project, please ensure that you have access."
                )
                full_path = custom_project
            else:
                if self.team_name == "wildlife-ai":
                    logging.info(
                        "Please note: Using models from adi-ohad-heb-uni account."
                    )
                    full_path = "adi-ohad-heb-uni/project-wildlife-ai"
                elif self.project_name == "template_project":
                    full_path = f"{self.team_name}/spyfish_aotearoa"
                else:
                    full_path = f"{self.team_name}/{self.project_name.lower()}"
            api = wandb.Api()
            try:
                api.artifact_type(type_name="model", project=full_path).collections()
            except Exception as e:
                logging.error(
                    f"No model collections found. No artifacts have been logged. {e}"
                )
                return None
            collections = [
                coll
                for coll in api.artifact_type(
                    type_name="model", project=full_path
                ).collections()
            ]
            model = [i for i in collections if i.name == model_name]
            if len(model) > 0:
                model = model[0]
            else:
                logging.error("No model found")
            artifact = api.artifact(full_path + "/" + model.name + ":latest")
            logging.info("Downloading model checkpoint...")
            artifact_dir = artifact.download(root=download_path)
            logging.info("Checkpoint downloaded.")
            return str(Path(artifact_dir).resolve())
        else:
            return

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
        if self.registry == "wandb":
            api = wandb.Api()
            if "_" in model:
                run_id = model.split("_")[1]
                try:
                    run = api.run(f"{team_name}/{self.project_name}/runs/{run_id}")
                except wandb.CommError:
                    logging.error("Run data not found")
                    return "", ""
                datasets = [
                    artifact
                    for artifact in run.used_artifacts()
                    if artifact.type == "dataset"
                ]
                if len(datasets) == 0:
                    logging.error(
                        "No datasets are linked to these runs. Please try another run."
                    )
                    return "", ""
                dirs = []
                for i in range(len(["train", "val"])):
                    artifact = datasets[i]
                    logging.info(f"Downloading {artifact.name} checkpoint...")
                    artifact_dir = artifact.download()
                    logging.info(f"{artifact.name} - Dataset downloaded.")
                    dirs.append(artifact_dir)
                return dirs
            else:
                logging.error("Externally trained model. No data available.")
                return "", ""
        else:
            logging.error("Unsupported registry")
            return "", ""
