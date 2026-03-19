
import torch
from torch.utils.data.sampler import WeightedRandomSampler, Sampler
from typing import Iterator, List, Sequence, Sized, Optional
import numpy as np


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


class BatchWeightedRandomProgSampler(Sampler):
    r"""Wraps weighted random sampler to yield a mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
        data_source (Dataset): dataset to sample from
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, batch_size: int, data_source: Sized, num_samples: Optional[int] = None,
                 replacement: bool = True, drop_last: bool = False, generator=None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self._num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        self.weights = torch.as_tensor(self.data_source.gen_weights(), dtype=torch.double)
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        for idx in range(len(rand_tensor) // self.batch_size):
            yield rand_tensor[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()
        if not self.drop_last and len(rand_tensor) % self.batch_size != 0:
            yield rand_tensor[(idx + 1) * self.batch_size:].tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
        
    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

class BatchRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        batch_size (int): batch size
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized

    def __init__(self, batch_size: int,  data_source: Sized,
                 num_samples: Optional[int] = None, drop_last: bool = False, generator=None) -> None:
        self.batch_size = batch_size
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator
        self.drop_last = drop_last

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        rand_tensor = torch.randperm(n, generator=generator)
        for idx in range(len(rand_tensor) // self.batch_size):
            yield rand_tensor[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()
        if not self.drop_last and len(rand_tensor) % self.batch_size != 0:
            yield rand_tensor[(idx + 1) * self.batch_size:].tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

class BatchWeightedRandomSampler(Sampler):
    r"""Wraps weighted random sampler to yield a mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, batch_size: int, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, drop_last: bool = False, generator=None) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        for idx in range(len(rand_tensor) // self.batch_size):
            yield rand_tensor[idx * self.batch_size:(idx + 1) * self.batch_size].tolist()
        if not self.drop_last and len(rand_tensor) % self.batch_size != 0:
            yield rand_tensor[(idx + 1) * self.batch_size:].tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size  # type: ignore[arg-type]
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]

class DynamicWeightedRandomSampler(BatchWeightedRandomSampler):
    def __init__(self, batch_size: int, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, drop_last: bool = False, generator=None) -> None:
        super().__init__(batch_size, weights, num_samples, replacement, drop_last, generator)
        self.initial_weights = weights

    def set_weights(self, new_weights: Sequence[float]):
        self.weights = torch.as_tensor(new_weights, dtype=torch.double)

    def reset_weights(self):
        self.weights = torch.as_tensor(self.initial_weights, dtype=torch.double)