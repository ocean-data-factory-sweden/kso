##### When u past the snippet in the textbox, make sure u remove the !pip install roboflow    ########
##### and that model = project.version(version_nr).model() is added at the end of the snippet ########

import sys
import os
import csv
import json
import pandas as pd
import folium
from PIL import Image
from shapely.geometry import Polygon
from roboflow import Roboflow
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView  
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, 
                             QLineEdit, QFormLayout, QProgressBar, QHBoxLayout)

class MapWindow(QWidget):
    def __init__(self, map_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seagrass Map")
        self.resize(1200, 800)  # Set a larger initial size for better usability

        layout = QVBoxLayout(self)

        # Display a placeholder label while the map loads
        self.map_label = QLabel("Loading map...")
        layout.addWidget(self.map_label)

        # Create a QWebEngineView instance for the map
        self.web_view = QWebEngineView(self)
        self.web_view.urlChanged.connect(self._on_url_changed)  # Connect to URL change event
        layout.addWidget(self.web_view)

        # Load the map
        self.web_view.load(QUrl.fromLocalFile(map_path))

    def _on_url_changed(self, url):
        # Update the label when the map has finished loading (optional)
        if url.scheme() == 'qrc':  # Check if it's a loading indicator URL
            self.map_label.setText("Map loaded.")

class SeafloorSegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Seafloor Segmentation')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Add documentation tooltips
        def create_input_with_tooltip(label_text, tooltip_text):
            hbox = QHBoxLayout()
            line_edit = QLineEdit(self)
            line_edit.setFixedHeight(30)
            line_edit.setFixedWidth(300)
            tooltip_label = QLabel('❓', self)
            tooltip_label.setToolTip(tooltip_text)
            hbox.addWidget(line_edit)
            hbox.addWidget(tooltip_label)
            form_layout.addRow(label_text, hbox)
            return line_edit

        self.snippet_input = create_input_with_tooltip('Snippet:', 'After training a model on Roboflow, you can copy the snippet from the "Version - export dataset" tab and paste it here. Before pasting the snippet within this textbox, remove the "!pip install roboflow" part from the snippet. If the model is not defined within the snippet, add model = project.version(1).model() at the end.')
        self.base_output_dir_input = create_input_with_tooltip('Base Output Directory:', 'Paste the path to the directory where you want to store the output files. A new directory will be created if it does not exist. If it does exist, a new directory will be created with an incremented number.')
        self.input_directory_input = create_input_with_tooltip('Input Directory:', 'Fill in the path to the directory containing the images you want to run the segmentation on.')
        self.confidence_threshold_input = create_input_with_tooltip('Confidence Threshold:', 'Set the confidence threshold for the predictions. Only predictions with a confidence score above this threshold will be saved.')
        self.confidence_threshold_input.setText('30')
        self.csv_path_input = create_input_with_tooltip('CSV File Path:', "Fill in the path to the CSV file containing the geolocation data. The CSV file should have a 'filename' column with the image filenames and a 'PhotoPosition' column with the geolocation data in the format 'latitude,longitude'.")

        layout.addLayout(form_layout)

        self.run_button = QPushButton('Run Segmentation', self)
        self.run_button.clicked.connect(self.run_segmentation)
        layout.addWidget(self.run_button)

        self.show_results_button = QPushButton('Show Results', self)
        self.show_results_button.clicked.connect(self.show_results)
        layout.addWidget(self.show_results_button)

        self.show_map_button = QPushButton('Show Map', self)
        self.show_map_button.clicked.connect(self.show_map)
        layout.addWidget(self.show_map_button)

        self.status_label = QLabel('Ready')
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def run_segmentation(self):
        snippet = self.snippet_input.text()
        base_output_dir = self.base_output_dir_input.text()
        input_directory = self.input_directory_input.text()
        confidence_threshold = int(self.confidence_threshold_input.text())
        csv_path = self.csv_path_input.text()

        if not snippet or not base_output_dir or not input_directory or not confidence_threshold or not csv_path:
            QMessageBox.warning(self, 'Error', 'All fields must be filled.')
            return

        self.status_label.setText('Running segmentation...')
        QApplication.processEvents()

        self.output_dir = self.create_output_dir(base_output_dir)

        # Set the progress bar range and reset value
        num_files = len([name for name in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, name))])
        self.progress_bar.setRange(0, num_files)
        self.progress_bar.setValue(0)

        self.run_processing(snippet, input_directory, self.output_dir, confidence_threshold, csv_path)

        self.status_label.setText('Segmentation completed.')

    def show_results(self):
        if not hasattr(self, 'output_dir'):
            QMessageBox.warning(self, 'Error', 'No output directory available. Run segmentation first.')
            return

        results_path = os.path.join(self.output_dir, 'results.csv')
        if os.path.exists(results_path):
            subprocess.call(['open', results_path])
        else:
            QMessageBox.warning(self, 'Error', 'Results file not found.')

    def show_map(self):
        if not hasattr(self, 'output_dir'):
            QMessageBox.warning(self, 'Error', 'No output directory available. Run segmentation first.')
            return

        map_path = os.path.join(self.output_dir, 'seagrass_map.html')
        if os.path.exists(map_path):
            # Create and show the MapWindow
            map_window = MapWindow(map_path, self)
            map_window.show()
        else:
            QMessageBox.warning(self, 'Error', 'Map file not found.')

    def create_output_dir(self, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            return base_dir
        else:
            counter = 2
            while True:
                new_dir = f"{base_dir}{counter}"
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    return new_dir
                counter += 1

    def run_processing(self, snippet, input_directory, output_dir, confidence_threshold, csv_path):
   
        local_vars = {}
        exec(snippet, globals(), local_vars)
        model = local_vars['model']

        for i, filename in enumerate(os.listdir(input_directory)):
            filepath = os.path.join(input_directory, filename)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                prediction = model.predict(filepath, confidence=confidence_threshold)
                prediction_data = prediction.json()
                csv_filename = os.path.splitext(filename)[0] + "_prediction.csv"
                csv_filepath = os.path.join(output_dir, csv_filename)

                with open(csv_filepath, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    header = ["x", "y", "width", "height", "confidence", "class", "class_id", "detection_id", "image_path", "prediction_type", "points", "area"]
                    csv_writer.writerow(header)

                    rows = []
                    for pred in prediction_data['predictions']:
                        mask_points = pred.get('points', [])
                        if mask_points:
                            mask_points = [(float(point['x']), float(point['y'])) for point in mask_points]
                            polygon = Polygon(mask_points)
                            area = polygon.area
                        else:
                            area = 0

                        row = {
                            "x": pred.get('x', ''),
                            "y": pred.get('y', ''),
                            "width": pred.get('width', ''),
                            "height": pred.get('height', ''),
                            "confidence": pred.get('confidence', ''),
                            "class": pred.get('class', ''),
                            "class_id": pred.get('class_id', ''),
                            "detection_id": pred.get('detection_id', ''),
                            "image_path": pred.get('image_path', ''),
                            "prediction_type": pred.get('prediction_type', ''),
                            "points": json.dumps(mask_points),
                            "area": area,
                        }
                        rows.append(row)

                    for row in rows:
                        csv_writer.writerow([row["x"], row["y"], row["width"], row["height"], row["confidence"], row["class"], row["class_id"], row["detection_id"], row["image_path"], row["prediction_type"], row["points"], row["area"]])

                labeled_prediction = model.predict(filepath, confidence=confidence_threshold)
                output_image_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_prediction.jpg")
                labeled_prediction.save(output_image_path)

                # Update progress bar
                self.progress_bar.setValue(i + 1)
                QApplication.processEvents()

        results, all_classes = self.calculate_area_percentages(input_directory, output_dir)
        results_csv_filepath = os.path.join(output_dir, 'results.csv')
        self.write_results_to_csv(results, all_classes, results_csv_filepath)
        self.getting_geolocation(output_dir, csv_path)

    def calculate_area_percentages(self, input_directory, output_dir):
        results = {}
        all_classes = set()

        for filename in os.listdir(output_dir):
            if filename.endswith('_prediction.csv'):
                csv_filepath = os.path.join(output_dir, filename)
                class_area_percentages = {}
                total_quadrant_area = 0
                total_image_area = 0
                non_quadrant_detections = False

                with open(csv_filepath, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        class_name = row['class']
                        if class_name == "quadrant":
                            width = float(row['width'])
                            height = float(row['height'])
                            total_quadrant_area = width * height
                            class_area_percentages[class_name] = total_quadrant_area
                            all_classes.add(class_name)
                            continue

                        area = float(row.get('area', 0))
                        if area > 0:
                            non_quadrant_detections = True

                        if class_name in class_area_percentages:
                            class_area_percentages[class_name] += area
                        else:
                            class_area_percentages[class_name] = area
                        all_classes.add(class_name)

                image_name = filename.replace('_prediction.csv', '')
                results[image_name] = {}
                if total_quadrant_area == 0:
                    image_path = os.path.join(input_directory, image_name + '.jpg')
                    with Image.open(image_path) as img:
                        total_image_area = img.width * img.height

                if non_quadrant_detections:
                    for class_name, total_area in class_area_percentages.items():
                        if class_name == "quadrant":
                            results[image_name][class_name] = 100.0
                        elif total_quadrant_area > 0:
                            area_percentage = (total_area / total_quadrant_area) * 100
                            results[image_name][class_name] = area_percentage
                        else:
                            area_percentage = (total_area / total_image_area) * 100
                            results[image_name][class_name] = area_percentage
                else:
                    results[image_name] = {'No detections besides the quadrant': 'Na'}

        return results, all_classes

    def write_results_to_csv(self, results, all_classes, output_filepath):
        formatted_results = {}
        for image_name, class_areas in results.items():
            formatted_results[image_name] = {class_name: class_areas.get(class_name, 'Na') for class_name in all_classes}

        df = pd.DataFrame.from_dict(formatted_results, orient='index', columns=sorted(all_classes))
        df.index.name = 'filename'
        df.to_csv(output_filepath, na_rep='Na')

    def getting_geolocation(self, output_dir, csv_path):
        photos_koster_df = pd.read_csv(csv_path)
        results_path = os.path.join(output_dir, 'results.csv')
        results_df = pd.read_csv(results_path)

        photos_koster_df = photos_koster_df.rename(columns={'PhotoPosition': 'PhotoPosition_koster'})

        if 'filename' not in photos_koster_df.columns or 'filename' not in results_df.columns:
            raise KeyError("'filename' column not found in one of the CSV files")

        photos_koster_df['filename'] = photos_koster_df['filename'].str.strip().str.lower()
        results_df['filename'] = results_df['filename'].str.strip().str.lower()

        results_df['filename'] = results_df['filename'].apply(lambda x: x if x.lower().endswith('.jpg') else x + '.jpg')

        merged_df = pd.merge(results_df, photos_koster_df[['filename', 'PhotoPosition_koster']], on='filename', how='left')

        missing_matches = merged_df[merged_df['PhotoPosition_koster'].isna()]
        if not missing_matches.empty:
            print("Filenames in results.csv with no matching PhotoPosition_koster:")
            print(missing_matches['filename'])

        merged_df['PhotoPosition'] = merged_df['PhotoPosition_koster']
        merged_df = merged_df.drop(columns=['PhotoPosition_koster'])

        merged_df.to_csv(results_path, index=False)

        map_center = [58.0, 11.0]
        m = folium.Map(location=map_center, zoom_start=8)

        for idx, row in merged_df.iterrows():
            if not pd.isna(row['PhotoPosition']):
                lat, lon = map(float, row['PhotoPosition'].split(','))
                color = 'green' if row['Seagrass'] else 'transparent'
                popup_text = 'Seagrass present' if row['Seagrass'] else 'No seagrass'

                folium.Rectangle(
                    bounds=[[lat-0.00007, lon-0.00007], [lat+0.00007, lon+0.00007]],
                    color='black',
                    fill=False,
                    fill_color=color,
                    fill_opacity=0.5 if row['Seagrass'] else 0,
                    tooltip=popup_text
                ).add_to(m)

        map_path = os.path.join(output_dir, 'seagrass_map.html')
        m.save(map_path)
        print(f'Map saved to {map_path}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SeafloorSegmentationApp()
    ex.show()
    sys.exit(app.exec_())
