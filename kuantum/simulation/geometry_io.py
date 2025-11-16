"""Utilities for importing detector geometries from external descriptions."""

from __future__ import annotations

import json
import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional

from .detector import DetectorGeometry, DetectorLayer


_GDML_UNIT_SCALES: Dict[str, float] = {
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
}

_MATERIAL_COLOURS: Dict[str, str] = {
    "Silicon": "#1f77b4",
    "Scintillator": "#ff7f0e",
    "Drift Tubes": "#2ca02c",
    "Lead": "#b0b0b0",
    "Argon": "#b58900",
    "Aluminium": "#7f8c8d",
}


def load_geometry(path: Path | str, *, fmt: Optional[str] = None, tree: Optional[str] = None) -> DetectorGeometry:
    """Load a :class:`DetectorGeometry` from a GDML/ROOT/JSON file."""

    file_path = Path(path)
    if fmt is None:
        fmt = file_path.suffix.lstrip(".")
    fmt = (fmt or "").lower()
    if fmt == "gdml":
        return _load_geometry_from_gdml(file_path)
    if fmt in {"json", "geom.json"}:
        return _load_geometry_from_json(file_path)
    if fmt in {"root", "geometry", "geo"}:
        return _load_geometry_from_root(file_path, tree=tree)
    raise ValueError(f"Unsupported geometry format '{fmt}'. Supported formats: gdml, json, root")


def _load_geometry_from_json(path: Path) -> DetectorGeometry:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        layers_data = data.get("layers")
    else:
        layers_data = data
    if not isinstance(layers_data, Iterable):
        raise ValueError("JSON geometry must define an iterable of layers")
    layers = []
    for entry in layers_data:
        material = entry.get("material", "Unknown")
        color = entry.get("color") or _MATERIAL_COLOURS.get(material, "white")
        layers.append(
            DetectorLayer(
                name=entry["name"],
                inner_radius=float(entry["inner_radius"]),
                outer_radius=float(entry["outer_radius"]),
                length=float(entry["length"]),
                material=material,
                color=color,
            )
        )
    layers.sort(key=lambda layer: layer.inner_radius)
    return DetectorGeometry(layers=layers)


def _load_geometry_from_gdml(path: Path) -> DetectorGeometry:
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0].strip("{")

    def _tag(name: str) -> str:
        return f"{{{namespace}}}{name}" if namespace else name

    solids = {element.attrib["name"]: element for element in root.findall(f".//{_tag('tube')}")}
    layers = []
    for volume in root.findall(f".//{_tag('volume')}"):
        solid_ref = volume.find(_tag("solidref"))
        if solid_ref is None:
            continue
        solid_name = solid_ref.attrib.get("ref")
        solid = solids.get(solid_name)
        if solid is None:
            continue
        unit = solid.attrib.get("lunit", "mm")
        scale = _GDML_UNIT_SCALES.get(unit, 1.0)
        try:
            inner_radius = float(solid.attrib.get("rmin", "0")) * scale
            outer_radius = float(solid.attrib["rmax"]) * scale
            length = float(solid.attrib.get("z", solid.attrib.get("dz", "0"))) * scale
        except KeyError:
            continue
        material_ref = volume.find(_tag("materialref"))
        material = material_ref.attrib.get("ref") if material_ref is not None else "Unknown"
        color = _MATERIAL_COLOURS.get(material, "#d0d0d0")
        name = volume.attrib.get("name", solid_name or f"Layer_{len(layers)}")
        layers.append(
            DetectorLayer(
                name=name,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                length=length,
                material=material,
                color=color,
            )
        )
    if not layers:
        raise ValueError("No cylindrical GDML volumes with <tube> solids were found")
    layers.sort(key=lambda layer: layer.inner_radius)
    return DetectorGeometry(layers=layers)


def _load_geometry_from_root(path: Path, *, tree: Optional[str] = None) -> DetectorGeometry:
    try:
        import uproot  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("The 'uproot' package is required to load ROOT geometries") from exc

    tree_name = tree or "detector_layers"
    with uproot.open(path) as handle:  # type: ignore[attr-defined]
        if tree_name not in handle:
            raise KeyError(
                f"ROOT file does not contain a '{tree_name}' tree. "
                "Provide --geometry-tree to select the correct dataset."
            )
        ttree = handle[tree_name]
        arrays = ttree.arrays(library="np")
    required = {"name", "inner_radius", "outer_radius", "length", "material"}
    missing = required.difference(arrays)
    if missing:
        raise KeyError(f"ROOT geometry is missing required branches: {', '.join(sorted(missing))}")
    layers = []
    names = arrays["name"]
    for index in range(len(names)):
        name = str(names[index])
        inner_radius = float(arrays["inner_radius"][index])
        outer_radius = float(arrays["outer_radius"][index])
        length = float(arrays["length"][index])
        material_value = arrays["material"][index]
        material = str(material_value) if not isinstance(material_value, str) else material_value
        if not math.isfinite(inner_radius) or not math.isfinite(outer_radius):
            warnings.warn(
                f"Skipping layer '{name}' due to non-finite radius values",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        color = _MATERIAL_COLOURS.get(material, "#e0e0e0")
        layers.append(
            DetectorLayer(
                name=name,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                length=length,
                material=material,
                color=color,
            )
        )
    if not layers:
        raise ValueError("ROOT geometry did not yield any valid detector layers")
    layers.sort(key=lambda layer: layer.inner_radius)
    return DetectorGeometry(layers=layers)


__all__ = ["load_geometry"]
