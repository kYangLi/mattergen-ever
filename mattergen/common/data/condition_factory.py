# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from itertools import accumulate
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import NumAtomsCrystalDataset
from mattergen.common.data.num_atoms_distribution import NUM_ATOMS_DISTRIBUTIONS
from mattergen.common.data.transform import SetProperty, Transform
from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_utils import create_chem_graph_from_composition
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.common.utils.globals import SELECTED_ATOMIC_SYMBOLS

ConditionLoader = Iterable[tuple[BatchedData, dict[str, torch.Tensor]] | None]


def randomly_select_elem(curr_x, exclude_x):
    elem_list = [
        v for v in SELECTED_ATOMIC_SYMBOLS
        if (v not in curr_x) and (v not in exclude_x)
    ]
    elem = elem_list[torch.randint(0, len(elem_list), (1,)).item()]
    return elem


def explain_elem_str(x: str) -> str:
    if '*' not in x:
        return x
    if '-' in x:
        new_x = x.split("-")
        new_x, x_w = new_x[:-1], new_x[-1]
    else:
        new_x, x_w = [], x
    if '!' in x_w:
        x_star, x_exclude = x_w.split("!")
        x_exclude = x_exclude.split(",")
    else:
        x_star, x_exclude = x_w, []
    if '%' in x_star:
        x_star, x_star_ratio = x_star.split("%")
        x_star_ratio = x_star_ratio.split(",")
        x_star_ratio = [int(v) for v in x_star_ratio]
    else:
        x_star_ratio = [1 for _ in x_star]
    x_star_ratio = [v/sum(x_star_ratio) for v in x_star_ratio]
    x_star_ratio = list(accumulate(x_star_ratio))
    _peek = torch.rand(1).item()
    max_other_elem_num = len(x_star)
    other_elem_num = min(
        sum([_peek > v for v in x_star_ratio]) + 1, max_other_elem_num
    )
    for _ in range(other_elem_num):
        random_elem = randomly_select_elem(new_x, x_exclude)
        new_x.append(random_elem)
    new_x = '-'.join(new_x)
    return new_x


def _collate_fn(
    batch: Sequence[ChemGraph],
    collate_fn: Callable[[Sequence[ChemGraph]], BatchedData],
) -> tuple[BatchedData, None]:
    if hasattr(batch[0], 'chemical_system'):
        for i, bi in enumerate(batch):
            batch[i] = bi.replace(chemical_system=explain_elem_str(bi.chemical_system))
    return collate_fn(batch), None


def get_number_of_atoms_condition_loader(
    num_atoms_distribution: str,
    num_samples: int,
    batch_size: int,
    shuffle: bool = True,
    transforms: list[Transform] | None = None,
    properties: TargetProperty | None = None,
) -> ConditionLoader:
    transforms = transforms or []
    if properties is not None:
        for k, v in properties.items():
            transforms.append(SetProperty(k, v))
    assert (
        num_atoms_distribution in NUM_ATOMS_DISTRIBUTIONS
    ), f"Invalid num_atoms_distribution: {num_atoms_distribution}"
    dataset = NumAtomsCrystalDataset.from_num_atoms_distribution(
        num_atoms_distribution=NUM_ATOMS_DISTRIBUTIONS[num_atoms_distribution],
        num_samples=num_samples,
        transforms=transforms,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=shuffle,
    )


def get_composition_data_loader(
    target_compositions_dict: list[dict[str, float]],
    num_structures_to_generate_per_composition: int,
    batch_size: int,
) -> ConditionLoader:
    """
    Given a list of target compositions, generate a dataset of chemgraphs
    where each chemgraph contains atoms corresponding to the target composition
    without positions or cell information.
    Returns a torch dataloader equipped with the correct collate function containing such dataset.
    """

    dataset_ = []
    for compostion in target_compositions_dict:
        chemgraphs = [
            create_chem_graph_from_composition(compostion)
        ] * num_structures_to_generate_per_composition
        dataset_.extend(chemgraphs)

    dataset = ChemGraphlistDataset(dataset_)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=partial(_collate_fn, collate_fn=collate),
        shuffle=False,
    )


class ChemGraphlistDataset(Dataset):
    def __init__(self, data: list[ChemGraph]) -> None:
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> ChemGraph:
        return self.data[index]
