"""Tests for the Darwin Core occurrence export (kso.publish_occurrences).

The tables are small and made up, so every expected number can be worked out
by hand. Taxonomy is passed in, so nothing is looked up in WoRMS.
"""

import hashlib
import json

import pandas as pd
import pytest

from kso.publish_occurrences import (
    PublicationConfig,
    build_events,
    format_to_gbif_occurrence,
    write_ipt_package,
)

VIDEO = "clip01"
FIRST_BIN = "test:clip01:000000-000010"
SECOND_BIN = "test:clip01:000010-000020"
TAXONOMY = {
    name: {"publishable": True, "scientificName": name, "vernacularName": name}
    for name in ("Fucus", "Gadus morhua")
}


def publication():
    return PublicationConfig(
        dataset_name="test dataset",
        dataset_prefix="test",
        deployments={
            VIDEO: {
                "event_start": "2024-06-12T10:00:00+00:00",
                "decimal_latitude": 57.9,
                "decimal_longitude": 11.5,
            }
        },
    )


def trimmed_clip():
    """A clip cut from a longer video: 10 fps, every 2nd frame analysed.

    It keeps the long video's timestamps (5.0 to 14.8 s) but its frame numbers
    start again at 0. Fucus covers 40% of each analysed frame, except frames
    10 to 28, where the model found nothing, so those frames have no row.
    """
    frames = [f for f in range(0, 100, 2) if not 10 <= f <= 28]
    return pd.DataFrame(
        {
            "video": VIDEO,
            "frame": frames,
            "timestamp_s": [5.0 + f / 10 for f in frames],
            "class_name": "Fucus",
            "coverage_frac": 0.4,
            "n_instances": 3,
        }
    )


def export(table, **kwargs):
    pub = publication()
    events = build_events(table, pub, bin_seconds=10, stride=2)
    occ = format_to_gbif_occurrence(table, events, pub, taxonomy=TAXONOMY, **kwargs)
    return events, occ


def test_cover_counts_empty_frames_in_a_trimmed_clip():
    _, occ = export(trimmed_clip())
    cover = occ.set_index("eventID")["organismQuantity"]
    # first bin: 25 frames analysed (5.0 to 9.8 s), Fucus in 15 of them at 40%
    assert cover[FIRST_BIN] == pytest.approx(15 * 40 / 25)
    # second bin: 25 frames analysed, Fucus in all of them
    assert cover[SECOND_BIN] == pytest.approx(40.0)


def test_edge_bins_count_only_the_seconds_the_clip_covers():
    events, _ = export(trimmed_clip())
    events = events.set_index("eventID")
    # the bins are 10 s wide, but the clip only covers 5.0 to 14.8 s
    assert events.loc[FIRST_BIN, "sampleSizeValue"] == pytest.approx(5.0)
    assert events.loc[SECOND_BIN, "sampleSizeValue"] == pytest.approx(4.8)
    assert events.loc[FIRST_BIN, "eventDate"].startswith("2024-06-12T10:00:05")


def test_no_individual_count_unless_asked():
    _, occ = export(trimmed_clip())
    assert (occ["individualCount"] == "").all()


def test_individual_count_is_the_most_seen_in_one_frame():
    fish = pd.DataFrame(
        {
            "video": VIDEO,
            "frame": [0, 2, 2, 4],
            "timestamp_s": [5.0, 5.2, 5.2, 5.4],
            "class_name": "Gadus morhua",
        }
    )
    _, occ = export(fish, count_individuals=True)
    assert occ["individualCount"].tolist() == [2]


def test_provenance_records_the_settings_and_the_source_table(tmp_path):
    table_path = tmp_path / "coverage_by_frame.csv"
    trimmed_clip().to_csv(table_path, index=False)
    events, occ = export(pd.read_csv(table_path))

    written = write_ipt_package(
        occ, events, tmp_path / "ipt", publication(), TAXONOMY, source=table_path
    )

    provenance = json.loads(written["provenance"].read_text(encoding="utf-8"))
    assert provenance["settings"]["bin_seconds"] == 10
    assert provenance["settings"]["stride"] == 2
    assert provenance["settings"]["quantity_basis"] == "event_mean"
    assert provenance["settings"]["empty_frames_counted"] is True
    expected = hashlib.sha256(table_path.read_bytes()).hexdigest()
    assert provenance["source"]["sha256"] == expected
