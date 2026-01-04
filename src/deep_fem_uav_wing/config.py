from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PIPELINE_VERSION = "v0.1.0"


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def geometry_dir(self) -> Path:
        return self.raw_dir / "geometry"

    @property
    def mesh_dir(self) -> Path:
        return self.raw_dir / "mesh"

    @property
    def fem_dir(self) -> Path:
        return self.raw_dir / "fem"


def get_paths(project_root: Path) -> Paths:
    return Paths(project_root=project_root)


