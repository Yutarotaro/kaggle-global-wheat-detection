"""Load optional training-only wheat images from a pinned archive."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from data import sha


def unpack_supplement(manifest_path, archives, fixed_split, resources):
    if len(archives) != 1:
        raise ValueError("Expected one pinned Mons supplement ZIP.")
    directory = Path(resources.enter_context(TemporaryDirectory(prefix="wheat-mons-")))
    return directory, load_supplement(manifest_path, archives[0], directory, fixed_split)


def load_supplement(manifest_path, archive, destination, fixed_split):
    spec = json.loads(manifest_path.read_text())
    if spec["split_sha256"] != sha(fixed_split) or spec["archive_sha256"] != sha(archive):
        raise ValueError("Training supplement or competition split does not match its manifest.")
    with ZipFile(archive) as bundle:
        annotation = "gwhd2021-mons-boxes.json"
        records = json.loads(bundle.read(annotation))
        if not records or len(records) != spec["images"] or any(
            not key.startswith("gwhd2021-mons-") or Path(key).name != key for key in records
        ):
            raise ValueError("Invalid supplement image IDs.")
        files = {f"train/{key}.jpg" for key in records} | {annotation}
        if set(bundle.namelist()) != files or len(bundle.namelist()) != len(files):
            raise ValueError("Supplement archive does not exactly match its annotations.")
        bundle.extractall(destination)
    return records
