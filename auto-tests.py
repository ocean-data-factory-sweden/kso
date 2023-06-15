# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:20:56 2023

@author: DiewertjeDekker


This auto-test does not test if everything displays what it should display. 
It mainly tests if everything still runs without giving any errors.

This test contains the test for all notebooks. 

"""

zoo_cred=['email', 'password']

#--------------------The first 2 cells of every notebook-----------------------
import os
import sys

# for when we are running it on git, the yalm file wil install the docker image. then we do need to go to the correct directory
#os.chdir("kso-object-detection/tutorials")

os.chdir("C:/Users/DiewertjeDekker/Documents/kso-object-detection") # This is for Diewertje locally


# https://stackoverflow.com/questions/15044447/how-do-i-unit-testing-my-gui-program-with-python-and-pyqt
# for if we want to test the widgets



## Import Python packages
#try:
if "kso_utils" not in sys.modules:
    sys.path.append("..")
    import kso_utils.kso_utils
    sys.modules["kso_utils"] = kso_utils.kso_utils
#except:
    
    #print("Installing latest version from PyPI...")
    #%pip install -q kso-utils


# Import required modules
import kso_utils.tutorials_utils as t_utils
import kso_utils.project_utils as p_utils
import kso_utils.widgets as kso_widgets
from kso_utils.project import ProjectProcessor
import kso_utils.server_utils as s_utils



#-----------------Initiating of the project structure -------------------------
# Initiate project's database
project = p_utils.find_project(project_name='Template project')
pp = ProjectProcessor(project)



#----------------Tutorial 1----------------------------------------------------
"""
Does not test if the changes that can be made manually to the df are applied
Does not test if we can preview the movies
Both widgets can't be run due to a runtime error when nothing gets selected.
"""

# display the map Tutorial 1
pp.map_sites()
# retrieve and display movies
pp.get_movie_info()
# pp.preview_media(), cannot be tested, runtime error 
# check species dataframe
species_sheet_df = pp.check_species_meta()

# --- Manually updata metadata (same for sites, movies and species)
sites_df, sites_range_rows, sites_range_columns = pp.select_meta_range(meta_key="sites")

sites_df_filtered, sites_sheet = kso_widgets.open_csv(
    df=sites_df, df_range_rows=('0', '5'), df_range_columns=()
)

sites_sheet_df = pp.view_meta_changes(df_filtered=sites_df_filtered, sheet=sites_sheet)

# pp.update_meta(sites_sheet_df, "sites") # cannot be tested, runtime error

# --- Automatic check of movies metadata
review_method = kso_widgets.choose_movie_review()
gpu_available = kso_widgets.gpu_select()
#pp.check_movies_meta(
#    review_method=review_method.value, gpu_available=gpu_available.result
#)

#-------------Tutorial 2-------------------------------------------------------
"""
This tutorial will be removed
"""

#-------------Tutorial 3-------------------------------------------------------
"""
...
""" 
# Connect to zooniverse # automatic pasword on gitlab
pp.get_zoo_info(zoo_cred=zoo_cred)

pp.check_movies_uploaded('movie_1.mp4')

# HOW TO TEST THE GENERATE_ZU_CLIPS???



#-------------Tutorial 4-------------------------------------------------------
"""
It will not be tested if the actually uploading to Zooniverse works, 
since we do not want to upload things to them all the time.
""" 
# pp.get_frames() # How to test? is with widgets
# pp.generate_zu_frames() # How to test this?
# input_folder = kso_widgets.choose_folder()
# output_folder = kso_widgets.choose_folder()
# # Generate suitable frames for upload by modifying initial frames
# pp.generate_custom_frames(
#     input_path=input_folder.selected,
#     output_path=output_folder.selected,
#     skip_start=120,
#     skip_end=120,
#     num_frames=10,
#     frames_skip=None,
# )
# t_utils.check_frame_size(frame_paths=pp.generated_frames["modif_frame_path"].unique())
# t_utils.compare_frames(df=pp.generated_frames)



#-------------Tutorial 5-------------------------------------------------------
"""
...
""" 



#-------------Tutorial 6-------------------------------------------------------
"""
...
""" 




#-------------Tutorial 7-------------------------------------------------------
"""
...
""" 




#-------------Tutorial 8-------------------------------------------------------
"""
...
""" 
pp.choose_workflows(False, zoo_cred=zoo_cred)

workflow_checks = {'Workflow name: #0': 'Development workflow',
 'Subject type: #0': 'clip',
 'Minimum workflow version: #0': 1.0}

class_df = pp.get_classifications(
    workflow_checks,
    pp.zoo_info["workflows"],
    workflow_checks["Subject type: #0"],
    pp.zoo_info["classifications"],
)





