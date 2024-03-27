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
fps real NULL,
duration datetime NULL,
sampling_start real NULL,
sampling_end real NULL,
author text NULL,
site_id text NULL,
fpath text NULL,
UNIQUE (filename),
FOREIGN KEY (site_id) REFERENCES sites (id)
);

CREATE TABLE IF NOT EXISTS photos
(
id integer PRIMARY KEY,
filename text NOT NULL,
PhotoPosition int NULL,
siteName text NULL,
SurveyID int NULL,
UNIQUE (filename)
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
subject_set_id varchar(255),
created_at datetime,
https_location text NULL,
movie_id integer NULL,
FOREIGN KEY (movie_id) REFERENCES movies (id)
);

CREATE TABLE IF NOT EXISTS species
(
id integer PRIMARY KEY,
commonName text NOT NULL,
scientificName text NOT NULL,
taxonRank text NOT NULL,
kingdom text NOT NULL,
UNIQUE (commonName)
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
x_position real NULL,
y_position real NULL,
width real NULL,
height real NULL,
subject_id integer,
UNIQUE(species_id, x_position, y_position, width, height, subject_id)
FOREIGN KEY (species_id) REFERENCES species (id),
FOREIGN KEY (subject_id) REFERENCES subjects (id)
);
"""
