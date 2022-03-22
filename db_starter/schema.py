sql = """CREATE TABLE IF NOT EXISTS sites
(
id integer PRIMARY KEY,
siteName text NULL,
decimalLatitude varchar(255) NULL,
decimalLongitude varchar(255) NULL,
geodeticDatum varchar(255) NULL,
countryCode varchar(255) NULL,
UNIQUE (siteName)
);

CREATE TABLE IF NOT EXISTS movies
(
id integer PRIMARY KEY,
filename text NOT NULL,
created_on datetime NULL,
fps integer NULL,
duration datetime NULL,
sampling_start integer NULL,
sampling_end integer NULL,
author text NULL,
site_id integer NULL,
fpath text NULL,
UNIQUE (filename),
FOREIGN KEY (site_id) REFERENCES sites (id)
); 

CREATE TABLE IF NOT EXISTS photos
(
id integer PRIMARY KEY,
filename text NOT NULL,
created_on datetime NULL,
site_id integer NULL,
fpath text NULL,
UNIQUE (filename),
FOREIGN KEY (site_id) REFERENCES sites (id)
); 


CREATE TABLE IF NOT EXISTS subjects
(
id integer PRIMARY KEY,
subject_type varchar(255) NULL,
filename text NULL,
clip_start_time datetime NULL,
clip_end_time datetime NULL,
frame_exp_sp_id integer NULL,
frame_number integer NULL,
workflow_id varchar(255) NULL,
subject_set_id varchar(255),
classifications_count integer NULL,
retired_at datetime NULL,
retirement_reason text NULL,
created_at datetime,
https_location text NULL,
movie_id integer NULL,
FOREIGN KEY (movie_id) REFERENCES movies (id)
);

CREATE TABLE IF NOT EXISTS species
(
id integer PRIMARY KEY,
label text NOT NULL,
scientificName text NOT NULL,
taxonRank text NOT NULL,
kingdom text NOT NULL,
UNIQUE (label)
);

CREATE TABLE IF NOT EXISTS agg_annotations_clip
(
id integer PRIMARY KEY AUTOINCREMENT,
species_id integer,
how_many integer,
first_seen integer,
subject_id integer,
UNIQUE(species_id, subject_id)
FOREIGN KEY (subject_id) REFERENCES subjects (id),
FOREIGN KEY (species_id) REFERENCES species (id)
);

CREATE TABLE IF NOT EXISTS agg_annotations_frame
(
id integer PRIMARY KEY AUTOINCREMENT,
species_id integer NULL,
x_position integer NULL,
y_position integer NULL,
width integer NULL,
height integer NULL,
subject_id integer,
UNIQUE(species_id, x_position, y_position, width, height, subject_id)
FOREIGN KEY (species_id) REFERENCES species (id),
FOREIGN KEY (subject_id) REFERENCES subjects (id)
);

CREATE TABLE IF NOT EXISTS models
(
id integer PRIMARY KEY AUTOINCREMENT,
config_file text NOT NULL,
conf_thres real NOT NULL,
img_size integer NOT NULL,
iou_thres real NOT NULL,
names_file text NOT NULL,
weights_file text NOT NULL
);

CREATE TABLE IF NOT EXISTS model_annotations
(
id integer PRIMARY KEY AUTOINCREMENT,
frame_number integer NULL,
model_id integer NULL,
species_id integer NULL,
movie_id integer NULL,
created_at datetime NULL,
x_position integer NULL,
y_position integer NULL,
width integer NULL,
height integer NULL,
confidence integer NULL, 
UNIQUE(frame_number, model_id, movie_id, species_id)
FOREIGN KEY (model_id) REFERENCES models (id),
FOREIGN KEY (movie_id) REFERENCES movies (id),
FOREIGN KEY (species_id) REFERENCES species (id)
);
"""
