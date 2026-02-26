from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, RelightDataset
from rfstudio.data.dataparser import IDRDataparser, MaskedBlenderDataparser, MaskedIDRDataparser
from rfstudio.engine.task import Task, TaskGroup

from rfstudio_ds.data import SyntheticDynamicMonocularCostumeDepthDataset, SyntheticDynamicMultiViewBlenderRGBADataset, CMUPanonicRGBADataset
from rfstudio_ds.data.dataparser import SyntheticDynamicMonocularCostumeDepthDataparser, SyntheticDynamicMultiviewBlenderRGBADataparser, SDFFlowRGBADataparser



@dataclass
class DynamicDepthSynthesis(Task):
    dataset: SyntheticDynamicMonocularCostumeDepthDataset = SyntheticDynamicMonocularCostumeDepthDataset(Path("data/dg-mesh/horse/"))

    output: Path = Path("data/dg-mesh/horse/dump")

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=SyntheticDynamicMonocularCostumeDepthDataparser)

@dataclass
class Temp(Task):
    dataset: SyntheticDynamicMonocularCostumeDepthDataset = SyntheticDynamicMonocularCostumeDepthDataset(path=...)

    output: Path = ...

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=SyntheticDynamicMonocularCostumeDepthDataparser)

@dataclass
class DMVS2IDR(Task):
    dataset: SyntheticDynamicMultiViewBlenderRGBADataset = SyntheticDynamicMultiViewBlenderRGBADataset(path=Path("data/ObjSel-Dyn/toy/"))

    output: Path = Path("data/ObjSel-Dyn/toy")

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=SyntheticDynamicMultiviewBlenderRGBADataparser)

@dataclass
class CMU2Blender(Task):
    dataset: CMUPanonicRGBADataset = CMUPanonicRGBADataset(path=Path("data/sdfflow/pizza1/"))
    output: Path = Path("data/sdfflow/pizza1")

    def run(self) -> None:
        self.dataset.to(self.device)
        self.dataset.dump(self.output, exist_ok=True, dataparser=SDFFlowRGBADataparser)


if __name__ == '__main__':
    TaskGroup(
        cmu2blender = CMU2Blender(cuda=0),
        dmvs2idr = DMVS2IDR(cuda=0),
        dynamic_depth_synthesis = DynamicDepthSynthesis(cuda=0),
        temp = Temp(cuda=0),
    ).run()
