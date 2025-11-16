"""VR/AR exporters for Kuantum collision events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .detector import DetectorGeometry
from .physics import CollisionEvent, Particle


@dataclass
class VRSceneExporter:
    geometry: DetectorGeometry

    def export_html(self, events: Iterable[CollisionEvent], path: Path | str) -> Path:
        output_path = Path(path)
        data = {
            "geometry": [
                {
                    "name": layer.name,
                    "inner_radius": layer.inner_radius,
                    "outer_radius": layer.outer_radius,
                    "length": layer.length,
                    "color": layer.color,
                }
                for layer in self.geometry.layers
            ],
            "events": [self._event_payload(event) for event in events],
        }
        html = _render_three_js_scene(data)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def _event_payload(self, event: CollisionEvent) -> dict:
        return {
            "id": event.event_id,
            "label": event.event_family,
            "model_prediction": event.model_prediction,
            "true_label": event.true_label,
            "particles": [self._particle_payload(particle) for particle in event.particles],
        }

    def _particle_payload(self, particle: Particle) -> dict:
        return {
            "name": particle.display_name,
            "layer": particle.detector_layer,
            "energy": particle.four_vector.energy,
            "eta": particle.eta,
            "phi": particle.phi,
            "points": self._sample_track_points(particle),
        }

    def _sample_track_points(self, particle: Particle, *, steps: int = 32) -> List[List[float]]:
        import numpy as np

        direction = np.array([particle.four_vector.px, particle.four_vector.py, particle.four_vector.pz], dtype=float)
        norm = np.linalg.norm(direction) + 1e-9
        direction /= norm
        length = 8.0 * (1.0 + min(abs(particle.eta), 2.5) / 2.5)
        base = np.zeros(3)
        points = [base.tolist()]
        for step in range(1, steps):
            t = step / (steps - 1)
            offset = direction * length * t
            points.append(offset.tolist())
        return points


def _render_three_js_scene(data: dict) -> str:
    payload = json.dumps(data)
    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Kuantum VR Scene</title>
  <style>
    body {{ margin: 0; background: #000; color: #fff; font-family: 'Source Sans Pro', sans-serif; }}
    #hud {{ position: absolute; top: 12px; left: 12px; max-width: 30vw; background: rgba(10, 10, 20, 0.6); padding: 12px; border-radius: 8px; }}
    canvas {{ display: block; }}
  </style>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.158/build/three.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.158/examples/jsm/webxr/VRButton.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/three@0.158/examples/jsm/controls/OrbitControls.js\"></script>
</head>
<body>
  <div id=\"hud\">Kuantum VR Scene - use VRButton to enter immersive mode.</div>
  <script>
    const data = {payload};
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000011);

    const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(25, 20, 18);

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.xr.enabled = true;
    document.body.appendChild(renderer.domElement);
    document.body.appendChild(THREE.VRButton.createButton(renderer));

    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0x8080ff, 0.6);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1.2);
    pointLight.position.set(30, 20, 30);
    scene.add(pointLight);

    function createLayer(layer) {{
      const geometry = new THREE.CylinderGeometry(layer.outer_radius, layer.outer_radius, layer.length, 96, 1, true);
      const material = new THREE.MeshBasicMaterial({{ color: layer.color || '#888888', wireframe: true, opacity: 0.35, transparent: true }});
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);
    }}

    function createTrack(points, colour) {{
      const positions = [];
      points.forEach(p => positions.push(p[0], p[1], p[2]));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({{ color: colour, linewidth: 2 }});
      const line = new THREE.Line(geometry, material);
      scene.add(line);
    }}

    function hexFromColour(colour) {{
      if (!colour) return 0xffffff;
      if (colour.startsWith('#')) return parseInt(colour.substring(1), 16);
      return parseInt(colour, 16);
    }}

    data.geometry.forEach(layer => createLayer(layer));
    const palette = [0xffb000, 0xd94f2a, 0x3a6ea5, 0xf5c04e, 0x6bb1c9, 0xb07aa1, 0x8c6c3a];
    let colourIndex = 0;
    data.events.forEach(event => {{
      event.particles.forEach(particle => {{
        const colour = palette[colourIndex % palette.length];
        colourIndex += 1;
        createTrack(particle.points, colour);
      }});
    }});

    function animate() {{
      renderer.setAnimationLoop(() => {{
        controls.update();
        renderer.render(scene, camera);
      }});
    }}

    animate();
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""


__all__ = ["VRSceneExporter"]
