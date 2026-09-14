# This file contains everything Vulture should ignore.
# So if we have checked the function/variable/method is actually used, we can put it here.
# Put them all as: (without the '#'

# drawBoxes  # unused function (kso_utils\frame_utils.py:19)

# kso.publish_occurrences: public API called from NB04 (GBIF export section)
load_publication_config  # unused function (src\kso\publish_occurrences.py)
suggest_aphia_ids  # unused function (src\kso\publish_occurrences.py)
build_events  # unused function (src\kso\publish_occurrences.py)
format_to_gbif_occurrence  # unused function (src\kso\publish_occurrences.py)
validate_occurrences  # unused function (src\kso\publish_occurrences.py)
write_ipt_package  # unused function (src\kso\publish_occurrences.py)
