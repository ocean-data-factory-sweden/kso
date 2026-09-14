"""Darwin Core occurrence export for KSO2.

Turns the per-frame tables from the inference notebook (NB03) into a Darwin
Core occurrence table for upload to a GBIF IPT. This replaces the KSO1
``kso_utils.widgets.format_to_gbif``, which read site, movie and species
metadata from the SQLite database. KSO2 has no database, so that metadata
comes from the project YAML (``publication:`` block) and from WoRMS via
``pyworms``.

Typical use, from NB04::

    pub = load_publication_config(project)
    taxonomy = resolve_taxonomy(pub, class_names)
    events = build_events(cov, pub, bin_seconds=30, stride=STRIDE)
    occ = format_to_gbif_occurrence(cov, events, pub, taxonomy)
    validate_occurrences(occ)
    write_ipt_package(occ, events, out_dir, pub, taxonomy)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Darwin Core terms, in the order the IPT likes to see them.
TAXON_TERMS = (
    "scientificName scientificNameID taxonRank kingdom phylum class order family genus"
).split()

EVENT_TERMS = """
    eventID parentEventID eventDate year month day samplingProtocol
    sampleSizeValue sampleSizeUnit samplingEffort decimalLatitude decimalLongitude
    geodeticDatum coordinateUncertaintyInMeters footprintWKT countryCode locality
    minimumDepthInMeters maximumDepthInMeters
""".split()

OCCURRENCE_TERMS = """
    occurrenceID eventID basisOfRecord occurrenceStatus eventDate year month day
    scientificName scientificNameID taxonRank kingdom phylum class order family genus
    vernacularName verbatimIdentification identificationRemarks
    identificationVerificationStatus identifiedBy identifiedByID recordedBy
    organismQuantity organismQuantityType individualCount
    decimalLatitude decimalLongitude geodeticDatum coordinateUncertaintyInMeters
    footprintWKT countryCode locality minimumDepthInMeters maximumDepthInMeters
    samplingProtocol sampleSizeValue sampleSizeUnit samplingEffort associatedMedia
    datasetName institutionCode collectionCode license
""".split()

REQUIRED_NON_EMPTY = """
    occurrenceID basisOfRecord occurrenceStatus scientificName eventDate
    decimalLatitude decimalLongitude
""".split()


# --------------------------------------------------------------------------
# Publication config: the YAML block that replaces the KSO1 sites/movies/species tables
# --------------------------------------------------------------------------


@dataclass
class PublicationConfig:
    """Everything the export needs that is not in the detections table."""

    dataset_name: str
    dataset_prefix: str
    institution_code: str = ""
    collection_code: str = ""
    license: str = "http://creativecommons.org/licenses/by/4.0/legalcode"
    recorded_by: str = ""
    country_code: str = ""
    locality: str = ""
    geodetic_datum: str = "WGS84"
    coordinate_uncertainty_m: Optional[float] = 30.0
    minimum_depth_m: Optional[float] = None
    maximum_depth_m: Optional[float] = None
    sampling_protocol: str = ""
    basis_of_record: str = "MachineObservation"
    # model provenance -> identifiedBy / identificationRemarks
    model_name: str = ""
    model_version: str = ""
    model_doi: str = ""
    confidence_threshold: Optional[float] = None
    # per-deployment metadata, keyed by the `video` value in the detections table
    deployments: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # class_name -> AphiaID, or None for morphotypes with no taxon
    class_taxonomy: Dict[str, Optional[int]] = field(default_factory=dict)
    # pre-resolved taxonomy so the export can run without network
    taxon_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def deployment(self, video: str) -> Dict[str, Any]:
        try:
            return self.deployments[video]
        except KeyError:
            raise KeyError(
                f"No deployment metadata for video {video!r}. Add it under "
                f"publication.deployments in the project YAML. "
                f"Known: {sorted(self.deployments)}"
            ) from None


def load_publication_config(source: Any) -> PublicationConfig:
    """Build a :class:`PublicationConfig` from a YAML path, a dict, or a Project."""
    if isinstance(source, PublicationConfig):
        return source
    if isinstance(source, Mapping):
        data = dict(source)
    else:
        path = (
            source
            if isinstance(source, (str, Path))
            else getattr(source, "Config_file_path", None)
        )
        if not path:
            raise TypeError(f"Cannot read publication metadata from {type(source)!r}")
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    pub = data.get("publication", data) or {}
    known = set(PublicationConfig.__dataclass_fields__)
    if unknown := set(pub) - known:
        logger.warning("Ignoring unknown publication keys: %s", sorted(unknown))
    kwargs = {k: v for k, v in pub.items() if k in known}
    kwargs.setdefault("dataset_name", data.get("project_name", "kso-dataset"))
    kwargs.setdefault("dataset_prefix", _slug(kwargs["dataset_name"]))
    return PublicationConfig(**kwargs)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


# --------------------------------------------------------------------------
# Taxonomy (WoRMS, via pyworms)
# --------------------------------------------------------------------------


def resolve_taxonomy(
    pub: PublicationConfig,
    class_names: Optional[Iterable[str]] = None,
    offline: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Map each model class to a WoRMS record, or to the reason it cannot be published.

    Classes with a null AphiaID (morphotypes such as "algal turf") and classes
    WoRMS cannot resolve come back with ``publishable=False`` so nothing is
    dropped silently.
    """
    names = list(class_names) if class_names is not None else list(pub.class_taxonomy)
    resolved = {}
    for name in names:
        if name in pub.taxon_cache:
            rec = {"vernacularName": name, **pub.taxon_cache[name]}
            rec.setdefault("publishable", bool(rec.get("scientificName")))
            resolved[name] = rec
            continue
        aphia_id = pub.class_taxonomy.get(name, "__missing__")
        if aphia_id == "__missing__":
            resolved[name] = _unpublishable(
                name, "class not listed in publication.class_taxonomy"
            )
        elif aphia_id in (None, "", "null"):
            resolved[name] = _unpublishable(
                name, "morphotype / non-taxonomic class (AphiaID deliberately null)"
            )
        elif offline:
            resolved[name] = _unpublishable(
                name, f"AphiaID {aphia_id} not in taxon_cache and offline=True"
            )
        else:
            resolved[name] = _worms_taxon(name, int(aphia_id))
    return resolved


def _unpublishable(name: str, reason: str) -> Dict[str, Any]:
    return {"publishable": False, "reason": reason, "vernacularName": name}


def _worms_taxon(name: str, aphia_id: int) -> Dict[str, Any]:
    """Fetch one AphiaRecord and flatten it to Darwin Core taxon terms."""
    try:
        rec = _pyworms().aphiaRecordByAphiaID(aphia_id)
    except Exception as exc:  # network
        logger.error("WoRMS lookup failed for %s (AphiaID %s): %s", name, aphia_id, exc)
        return _unpublishable(name, f"WoRMS lookup failed: {exc}")
    if not rec:
        return _unpublishable(name, f"AphiaID {aphia_id} not found in WoRMS")

    valid_id = rec.get("valid_AphiaID") or aphia_id
    if rec.get("status") != "accepted" and rec.get("valid_name"):
        logger.warning(
            "AphiaID %s (%s) is %s; publishing the accepted name %s (AphiaID %s)",
            aphia_id,
            rec.get("scientificname"),
            rec.get("status"),
            rec["valid_name"],
            valid_id,
        )
    taxon = {
        "publishable": True,
        "vernacularName": name,
        "AphiaID": valid_id,
        "scientificName": rec.get("valid_name") or rec.get("scientificname"),
        # built from the accepted id: the record's own lsid points at the synonym
        "scientificNameID": f"urn:lsid:marinespecies.org:taxname:{valid_id}",
        "taxonRank": rec.get("rank"),
    }
    taxon.update({t: rec.get(t) for t in TAXON_TERMS[3:]})
    return taxon


def _pyworms():
    try:
        import pyworms
    except ImportError as exc:
        raise ImportError(
            "pyworms is needed for WoRMS lookups: "
            "pip install git+https://github.com/iobis/pyworms.git "
            "(or fill publication.taxon_cache and pass offline=True)"
        ) from exc
    return pyworms


def suggest_aphia_ids(class_names: Iterable[str]) -> pd.DataFrame:
    """Fuzzy-match class names against WoRMS (TAXAMATCH) to help fill ``class_taxonomy``.

    Never writes to the YAML. Read every row: a wrong AphiaID publishes the
    wrong species under your name.
    """
    names = list(class_names)
    try:
        matches = _pyworms().aphiaRecordsByMatchNames(
            [_strip_open_nomenclature(n) for n in names], marine_only=False
        )
    except Exception as exc:
        logger.warning("WoRMS match failed: %s", exc)
        matches = [[] for _ in names]

    rows = []
    for name, hits in zip(names, matches):
        if not hits:
            rows.append(
                {
                    "class_name": name,
                    "AphiaID": None,
                    "match": "NO MATCH",
                    "match_type": "",
                    "rank": "",
                    "status": "",
                    "note": "morphotype? leave null",
                }
            )
        for hit in (hits or [])[:3]:
            rows.append(
                {
                    "class_name": name,
                    "AphiaID": hit.get("valid_AphiaID") or hit.get("AphiaID"),
                    "match": hit.get("valid_name") or hit.get("scientificname"),
                    "match_type": hit.get("match_type", ""),
                    "rank": hit.get("rank"),
                    "status": hit.get("status"),
                    "note": hit.get("authority", ""),
                }
            )
    return pd.DataFrame(rows)


def _strip_open_nomenclature(name: str) -> str:
    """'Fucus sp.' -> 'Fucus'. Anchored on whitespace: the dot defeats a ``\\b``."""
    cleaned = re.sub(
        r"(?:^|\s)(?:sp|spp|cf|aff|indet)\.?(?=\s|$)", " ", name, flags=re.I
    )
    return re.sub(r"\s+", " ", cleaned).strip()


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


def build_events(
    df: pd.DataFrame,
    pub: PublicationConfig,
    bin_seconds: Optional[float] = None,
    video_col: str = "video",
    time_col: str = "timestamp_s",
    fps: Optional[float] = None,
    stride: Optional[int] = None,
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Cut each deployment into sampling events, each with a time and a position.

    ``bin_seconds=None`` gives one event per video; a number cuts the transect
    into fixed time bins, each with its own interpolated position when the
    deployment has a GPS track.

    ``stride`` (from the inference run) lets the number of frames actually
    sampled per event be reconstructed, which is the denominator for mean
    cover. Without it, frames where the model detected nothing - which produce
    no rows - drop out of the average and cover is overstated. ``fps`` is only
    needed when the table has no frame column.
    """
    _require_columns(df, [video_col, time_col], "detections/coverage table")
    rows = []
    for video, group in df.groupby(video_col, sort=True):
        video = str(video)
        meta = pub.deployment(video)
        start = _parse_datetime(meta.get("event_start"), video)
        times = pd.to_numeric(group[time_col], errors="coerce").dropna()
        if times.empty:
            logger.warning("Video %s has no usable %s values; skipped", video, time_col)
            continue
        t_min, t_max = float(times.min()), float(times.max())
        edges = (
            [t_min, t_max]
            if bin_seconds is None
            else _bin_edges(t_min, t_max, float(bin_seconds))
        )
        grid = _sampled_frame_times(group, t_min, t_max, fps, stride, frame_col, video)
        parent = f"{pub.dataset_prefix}:{video}"

        # per-deployment constants, falling back to the YAML-level defaults
        fixed = {
            "parentEventID": "" if bin_seconds is None else parent,
            "video": video,
            "samplingProtocol": meta.get("sampling_protocol", pub.sampling_protocol),
            "sampleSizeUnit": "seconds of video reviewed",
            "geodeticDatum": pub.geodetic_datum,
            "footprintWKT": meta.get("footprint_wkt", ""),
            "countryCode": meta.get("country_code", pub.country_code),
            "locality": meta.get("locality", pub.locality),
            "minimumDepthInMeters": meta.get("minimum_depth_m", pub.minimum_depth_m),
            "maximumDepthInMeters": meta.get("maximum_depth_m", pub.maximum_depth_m),
            "associatedMedia": meta.get("associated_media", ""),
        }
        for b0, b1 in zip(edges[:-1], edges[1:]):
            lat, lon, uncertainty = _position_for(meta, b0, b1, pub)
            e0 = start + timedelta(seconds=b0) if start else None
            e1 = start + timedelta(seconds=b1) if start else None
            n_sampled = (
                max(1, int(((grid >= b0) & (grid < b1)).sum()))
                if grid is not None
                else None
            )
            rows.append(
                {
                    **fixed,
                    "eventID": (
                        parent
                        if bin_seconds is None
                        else f"{parent}:{int(round(b0)):06d}-{int(round(b1)):06d}"
                    ),
                    "bin_start_s": b0,
                    "bin_end_s": b1,
                    "eventDate": _iso_interval(e0, e1),
                    "year": e0.year if e0 else "",
                    "month": e0.month if e0 else "",
                    "day": e0.day if e0 else "",
                    "sampleSizeValue": round(b1 - b0, 3),
                    "samplingEffort": (
                        f"{n_sampled} frames analysed (every {stride} frame(s) of video)"
                        if n_sampled
                        else ""
                    ),
                    "nFramesSampled": n_sampled,
                    "decimalLatitude": lat,
                    "decimalLongitude": lon,
                    "coordinateUncertaintyInMeters": uncertainty,
                }
            )

    if not rows:
        raise ValueError(
            "No events were built - check the video column and deployment metadata."
        )
    return pd.DataFrame(rows)


def _sampled_frame_times(group, t_min, t_max, fps, stride, frame_col, video):
    """Timestamps of every frame the model looked at, detections or not.

    Built from the frame column when there is one, with the effective frame
    rate measured from the data. That avoids the nominal-vs-actual fps trap:
    GoPro footage labelled 30 fps is really 29.97, and assuming 30 inflates
    every event's frame count by ~0.1%.
    """
    if not stride:
        return None
    if frame_col in group.columns:
        frames = pd.to_numeric(group[frame_col], errors="coerce").dropna()
        if not frames.empty and t_max > t_min:
            fps_effective = (frames.max() - frames.min()) / (t_max - t_min)
            if fps_effective > 0:
                return (
                    np.arange(0, int(frames.max()) + int(stride), int(stride))
                    / fps_effective
                )
    if fps:
        interval = float(stride) / float(fps)
        return np.arange(t_min, t_max + interval, interval)
    logger.warning(
        "Video %s: no frame column and no fps, so the sampled-frame grid is unknown "
        "and mean cover will be averaged over detected frames only.",
        video,
    )
    return None


def _bin_edges(t_min: float, t_max: float, size: float) -> List[float]:
    if size <= 0:
        raise ValueError("bin_seconds must be > 0")
    edges = [(t_min // size) * size]
    while edges[-1] < t_max:
        edges.append(edges[-1] + size)
    return edges


def _position_for(meta, b0, b1, pub):
    """GPS track midpoint if the deployment has one, else its fixed position."""
    uncertainty = meta.get("coordinate_uncertainty_m", pub.coordinate_uncertainty_m)
    if track := meta.get("gps_track"):
        lat, lon = _interpolate_track(track, (b0 + b1) / 2.0)
        return lat, lon, meta.get("gps_uncertainty_m", uncertainty)
    lat, lon = meta.get("decimal_latitude"), meta.get("decimal_longitude")
    if lat is None or lon is None:
        raise KeyError(
            "A deployment has neither a gps_track nor decimal_latitude/decimal_longitude. "
            "GBIF will not accept occurrences without coordinates."
        )
    return float(lat), float(lon), uncertainty


def _interpolate_track(track: Sequence[Mapping[str, float]], seconds: float):
    """Linear interpolation along a [{t, lat, lon}, ...] track, clamped at the ends."""
    pts = sorted((float(p["t"]), float(p["lat"]), float(p["lon"])) for p in track)
    ts = np.array([p[0] for p in pts])
    return (
        float(np.interp(seconds, ts, [p[1] for p in pts])),
        float(np.interp(seconds, ts, [p[2] for p in pts])),
    )


def _parse_datetime(value, video: str):
    if value in (None, ""):
        logger.warning(
            "Deployment %s has no event_start; eventDate will be empty and GBIF "
            "will reject or downgrade the records.",
            video,
        )
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Deployment {video}: event_start={value!r} is not ISO 8601 "
            f"(expected e.g. 2024-06-12T10:15:00+02:00)"
        ) from exc


def _iso_interval(start, end) -> str:
    if start is None:
        return ""
    return (
        start.isoformat()
        if end in (None, start)
        else f"{start.isoformat()}/{end.isoformat()}"
    )


# --------------------------------------------------------------------------
# Occurrences
# --------------------------------------------------------------------------


def format_to_gbif_occurrence(
    df: pd.DataFrame,
    events: pd.DataFrame,
    pub: PublicationConfig,
    taxonomy: Optional[Dict[str, Dict[str, Any]]] = None,
    quantity_col: Optional[str] = None,
    quantity_type: Optional[str] = None,
    video_col: str = "video",
    time_col: str = "timestamp_s",
    class_col: str = "class_name",
    confidence_col: str = "confidence",
    min_frames: int = 1,
    quantity_basis: str = "event_mean",
    offline: bool = False,
) -> pd.DataFrame:
    """One Darwin Core record per (event x class) the model saw.

    Coverage tables carry ``organismQuantity`` as mean percentage cover over
    the event; detection tables carry ``individualCount`` as the peak number
    of instances in any one frame.

    ``quantity_basis="event_mean"`` (default) averages over every frame
    sampled, counting frames without the class as zero - the figure that
    belongs next to ``sampleSizeValue``. ``"present_mean"`` averages only over
    frames where the class was seen; that conditional cover overstates
    abundance for the event and should not be published as organismQuantity.
    """
    if quantity_basis not in ("event_mean", "present_mean"):
        raise ValueError(
            f"quantity_basis must be 'event_mean' or 'present_mean', got {quantity_basis!r}"
        )
    _require_columns(df, [video_col, time_col, class_col], "detections/coverage table")

    work = df.copy()
    work[video_col] = work[video_col].astype(str)
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    quantity_col, quantity_type = _resolve_quantity(work, quantity_col, quantity_type)
    if taxonomy is None:
        taxonomy = resolve_taxonomy(
            pub, sorted(work[class_col].unique()), offline=offline
        )

    assigned = _assign_events(work, events, video_col, time_col)
    # Peak instances in any one frame: the coverage table counts them; a
    # detections table has one row per instance, so count rows per frame.
    if "n_instances" in assigned:
        assigned["__instances"] = pd.to_numeric(
            assigned["n_instances"], errors="coerce"
        )
    else:
        assigned["__instances"] = assigned.groupby(["eventID", class_col, time_col])[
            time_col
        ].transform("size")

    # nunique, not size: a detections table has several rows per frame
    aggs = {"n_frames": (time_col, "nunique"), "peak_instances": ("__instances", "max")}
    if quantity_col:
        aggs.update(
            quantity_sum=(quantity_col, "sum"),
            quantity_present_mean=(quantity_col, "mean"),
        )
    if confidence_col in assigned:
        aggs["conf_mean"] = (confidence_col, "mean")
    grouped = assigned.groupby(["eventID", class_col], as_index=False).agg(**aggs)
    grouped = grouped[grouped["n_frames"] >= int(min_frames)]
    merged = grouped.merge(events, on="eventID", how="left", validate="many_to_one")
    merged = _add_event_mean_quantity(
        merged, assigned, events, time_col, quantity_basis
    )

    records, dropped = [], {}
    for row in merged.to_dict("records"):
        name = row[class_col]
        taxon = taxonomy.get(name, {})
        if not taxon.get("publishable"):
            dropped.setdefault(name, taxon.get("reason", "no taxonomic identifier"))
            continue
        records.append(_occurrence_row(row, taxon, name, pub, quantity_type))
    if dropped:
        logger.warning(
            "Excluded %d class(es) with no taxonomic identifier: %s",
            len(dropped),
            "; ".join(f"{k} ({v})" for k, v in sorted(dropped.items())),
        )

    occ = pd.DataFrame(records, columns=OCCURRENCE_TERMS)
    if occ.empty:
        logger.error(
            "No publishable occurrences - fill in publication.class_taxonomy "
            "with AphiaIDs for the taxa you can name."
        )
        return occ
    if occ["occurrenceID"].duplicated().any():
        raise ValueError("Duplicate occurrenceIDs generated - this is a bug.")
    occ = occ.astype(object).where(occ.notna(), "")
    return occ.sort_values(["eventID", "scientificName"]).reset_index(drop=True)


def _occurrence_row(row, taxon, class_name, pub, quantity_type) -> Dict[str, Any]:
    quantity, count = row.get("quantity_mean"), row.get("peak_instances")
    remarks = [
        f"Automated identification by {pub.model_name or 'a KSO model'}"
        + (f" (version {pub.model_version})" if pub.model_version else ""),
        (
            f"confidence threshold {pub.confidence_threshold}"
            if pub.confidence_threshold is not None
            else ""
        ),
        (
            f"mean detection confidence {row['conf_mean']:.3f}"
            if pd.notna(row.get("conf_mean"))
            else ""
        ),
        (
            f"class detected in {int(row['n_frames'])} of {int(row['nFramesSampled'])} sampled frames"
            if pd.notna(row.get("nFramesSampled"))
            else f"class detected in {int(row['n_frames'])} sampled frame(s)"
        ),
        f"model label {class_name!r}",
    ]
    rec = {t: row.get(t, "") for t in EVENT_TERMS}  # where and when, from the event
    rec.update({t: taxon.get(t, "") for t in TAXON_TERMS})  # what, from WoRMS
    rec.update(
        occurrenceID=f"{row['eventID']}:{_slug(class_name)}",
        basisOfRecord=pub.basis_of_record,
        occurrenceStatus="present",
        vernacularName=taxon.get("vernacularName", class_name),
        verbatimIdentification=class_name,
        identificationRemarks="; ".join(r for r in remarks if r),
        identificationVerificationStatus="unverified - automated identification",
        identifiedBy=pub.model_name,
        identifiedByID=pub.model_doi,
        recordedBy=pub.recorded_by,
        organismQuantity=round(float(quantity), 6) if pd.notna(quantity) else "",
        organismQuantityType=quantity_type if pd.notna(quantity) else "",
        individualCount=int(count) if pd.notna(count) else "",
        associatedMedia=row.get("associatedMedia", ""),
        datasetName=pub.dataset_name,
        institutionCode=pub.institution_code,
        collectionCode=pub.collection_code,
        license=pub.license,
    )
    return rec


def _add_event_mean_quantity(merged, assigned, events, time_col, quantity_basis):
    """Set ``quantity_mean``, zero-filling frames where the class was absent.

    The denominator is the number of frames *sampled* in the event, from the
    stride grid recorded by :func:`build_events`. Frames where the model
    detected nothing produce no rows, so counting only observed frames would
    shrink the denominator and overstate cover.
    """
    if "quantity_sum" not in merged:
        merged["quantity_mean"] = pd.NA
        return merged
    if quantity_basis == "present_mean":
        merged["quantity_mean"] = merged["quantity_present_mean"]
        return merged

    observed = merged["eventID"].map(assigned.groupby("eventID")[time_col].nunique())
    if "nFramesSampled" in events and events["nFramesSampled"].notna().any():
        from_grid = merged["eventID"].map(events.set_index("eventID")["nFramesSampled"])
        # the grid is authoritative, but never fewer frames than we actually saw
        denominator = from_grid.fillna(observed).clip(lower=observed)
    else:
        logger.warning(
            "build_events was given no stride, so mean cover is averaged over the "
            "frames present in the table; frames where nothing was detected are "
            "missing from that denominator, which overstates cover."
        )
        denominator = observed
    merged["quantity_mean"] = merged["quantity_sum"] / denominator
    return merged


def _resolve_quantity(df, quantity_col, quantity_type):
    """Only ``coverage_frac`` becomes organismQuantity automatically.

    The segmentation post-processing already merges overlapping instances of a
    class, so it is a true per-class share of the frame. ``area_frac`` from
    the detections table is deliberately not used: boxes of the same class
    overlap, so summing them overstates cover. Detection tables get
    individualCount instead; pass ``quantity_col`` explicitly to override.
    """
    if quantity_col is not None:
        return quantity_col, quantity_type or "unspecified"
    if "coverage_frac" in df:
        df["__pct_cover"] = pd.to_numeric(df["coverage_frac"], errors="coerce") * 100.0
        return "__pct_cover", quantity_type or "percentage cover"
    return None, quantity_type


def _assign_events(df, events, video_col, time_col):
    """Attach each frame row to the event whose time bin contains it."""
    out = []
    for video, group in df.groupby(video_col, sort=False):
        video_events = events[events["video"] == str(video)].sort_values("bin_start_s")
        if video_events.empty:
            logger.warning("No events for video %s; its rows are skipped", video)
            continue
        ids = video_events["eventID"].tolist()
        edges = video_events["bin_start_s"].tolist() + [
            float(video_events["bin_end_s"].iloc[-1])
        ]
        codes = pd.cut(
            group[time_col], pd.IntervalIndex.from_breaks(edges, closed="left")
        ).cat.codes
        assigned = group.copy()
        # code -1 is the closing edge (the very last timestamp): keep it in the last bin
        assigned["eventID"] = [ids[c] if c >= 0 else ids[-1] for c in codes]
        out.append(assigned)
    if not out:
        raise ValueError("No rows could be assigned to an event.")
    return pd.concat(out, ignore_index=True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    if missing := [c for c in columns if c not in df.columns]:
        raise KeyError(
            f"{what} is missing column(s) {missing}. Found: {list(df.columns)}"
        )


# --------------------------------------------------------------------------
# Validation and packaging
# --------------------------------------------------------------------------


def validate_occurrences(occ: pd.DataFrame, strict: bool = False) -> List[str]:
    """Check what GBIF and OBIS actually require; return the list of problems."""
    problems: List[str] = []
    if occ.empty:
        problems.append("The occurrence table is empty.")
    else:
        for term in REQUIRED_NON_EMPTY:
            if term not in occ:
                problems.append(f"Missing required term {term}")
            elif n := _n_blank(occ[term]):
                problems.append(f"{term} is empty in {n}/{len(occ)} records")
        if occ["occurrenceID"].duplicated().any():
            problems.append("occurrenceID is not unique")
        for term, lo, hi in (
            ("decimalLatitude", -90, 90),
            ("decimalLongitude", -180, 180),
        ):
            values = pd.to_numeric(
                occ.get(term, pd.Series(dtype=float)), errors="coerce"
            )
            if values.notna().any():
                if n := int(((values < lo) | (values > hi)).sum()):
                    problems.append(f"{term} out of range in {n} records")
                if (values.abs() < 1e-9).all():
                    problems.append(
                        f"{term} is 0 everywhere - placeholder coordinates?"
                    )
        if "scientificNameID" in occ and (n := _n_blank(occ["scientificNameID"])):
            problems.append(
                f"scientificNameID (the WoRMS LSID) is empty in {n} records "
                f"- OBIS requires it; GBIF only recommends it"
            )
        if "countryCode" in occ and (n := _n_blank(occ["countryCode"])):
            problems.append(f"countryCode is empty in {n} records")

    if problems:
        message = "Occurrence table has issues:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ValueError(message)
        logger.warning(message)
    else:
        logger.info("Occurrence table passed validation.")
    return problems


def _n_blank(series: pd.Series) -> int:
    return int((series.isna() | (series.astype(str).str.strip() == "")).sum())


def write_ipt_package(
    occ: pd.DataFrame,
    events: Optional[pd.DataFrame],
    out_dir: str | Path,
    pub: Optional[PublicationConfig] = None,
    taxonomy: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    """Write the tab-delimited source files to upload to the IPT.

    Not a finished Darwin Core Archive: the IPT builds that itself once the
    columns are mapped in its UI, and keeps the mapping and EML editable.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {"occurrence": out / "occurrence.txt"}
    occ.to_csv(written["occurrence"], sep="\t", index=False, encoding="utf-8")
    if events is not None and not events.empty:
        written["event"] = out / "event.txt"
        events[[c for c in EVENT_TERMS if c in events]].to_csv(
            written["event"], sep="\t", index=False, encoding="utf-8"
        )
    provenance = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "n_occurrences": int(len(occ)),
        "n_events": int(len(events)) if events is not None else 0,
        "publication": (
            {k: v for k, v in asdict(pub).items() if k != "taxon_cache"} if pub else {}
        ),
        "taxonomy": taxonomy or {},
    }
    written["provenance"] = out / "provenance.json"
    written["provenance"].write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Wrote IPT source files to %s: %s",
        out,
        ", ".join(p.name for p in written.values()),
    )
    return written
