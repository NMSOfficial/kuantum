"""Physics-inspired reporting utilities for the Kuantum simulation."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .detector import DetectorGeometry
from .physics import CLASS_LABELS, CollisionEvent


@dataclass
class SimulationReporter:
    geometry: DetectorGeometry
    compile_pdf: bool = False

    def build_report(self, events: Iterable[CollisionEvent], output_path: Path | str) -> Path:
        output = Path(output_path)
        tex_source = self._render_latex(events)
        output.write_text(tex_source, encoding="utf-8")
        if self.compile_pdf:
            self._try_compile(output)
        return output

    def _render_latex(self, events: Iterable[CollisionEvent]) -> str:
        body = [
            r"\documentclass[11pt]{article}",
            r"\usepackage{geometry}",
            r"\usepackage{longtable}",
            r"\usepackage{booktabs}",
            r"\geometry{margin=1in}",
            r"\title{Kuantum Collision Run Report}",
            r"\begin{document}",
            r"\maketitle",
            r"\section*{Detector Geometry}",
            r"\begin{longtable}{lllll}",
            r"\toprule",
            r"Layer & $r_{\text{inner}}$ [m] & $r_{\text{outer}}$ [m] & Length [m] & Material\\",
            r"\midrule",
        ]
        for layer in self.geometry.layers:
            body.append(
                rf"{layer.name} & {layer.inner_radius:.2f} & {layer.outer_radius:.2f} & {layer.length:.2f} & {layer.material}\\"
            )
        body.extend([r"\bottomrule", r"\end{longtable}"])
        body.append(r"\section*{Event Catalogue}")
        body.append(r"\begin{longtable}{lllll}")
        body.append(r"\toprule")
        body.append(r"Event ID & Family & Model & Truth & $\Sigma E$ [GeV]\\")
        body.append(r"\midrule")
        for index, event in enumerate(events):
            total_energy = sum(particle.four_vector.energy for particle in event.particles)
            model = CLASS_LABELS.get(event.model_prediction, "?")
            truth = CLASS_LABELS.get(event.true_label, "?")
            body.append(rf"{event.event_id} & {event.event_family} & {model} & {truth} & {total_energy:.1f}\\")
            if index >= 63:
                body.append(r"\midrule")
                body.append(r"\multicolumn{5}{c}{\textit{Catalogue truncated for brevity}}\\")
                break
        body.append(r"\bottomrule")
        body.append(r"\end{longtable}")
        body.append(r"\end{document}")
        return "\n".join(body)

    def _try_compile(self, tex_path: Path) -> None:
        try:
            subprocess.run(
                ["pdflatex", tex_path.name],
                cwd=tex_path.parent,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError("pdflatex executable not found; disable compile_pdf or install TeX distribution")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"pdflatex failed: {exc.stderr.decode('utf-8', errors='ignore')}")


__all__ = ["SimulationReporter"]
