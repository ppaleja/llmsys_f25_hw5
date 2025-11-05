import pytest
from pipeline.pipe import _clock_cycles, Pipe
from pipeline.partition import _split_module, WithDevice
from torch import nn
import torch

# Skip the entire module if MPS is not available on this machine
if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    pytest.skip(
        "MPS backend is not available, skipping MPS-specific pipeline tests",
        allow_module_level=True,
    )


@pytest.mark.a5_2_1
def test_clock_cycles_0():
    m = 6
    n = 2
    gold_schedule = iter(
        [
            [(0, 0)],
            [(1, 0), (0, 1)],
            [(2, 0), (1, 1)],
            [(3, 0), (2, 1)],
            [(4, 0), (3, 1)],
            [(5, 0), (4, 1)],
            [(5, 1)],
        ]
    )
    for schedule in _clock_cycles(m, n):
        assert sorted(schedule) == sorted(next(gold_schedule))


@pytest.mark.a5_2_1
def test_clock_cycles_1():
    m = 3
    n = 3
    gold_schedule = iter(
        [
            [(0, 0)],
            [(1, 0), (0, 1)],
            [(2, 0), (1, 1), (0, 2)],
            [(2, 1), (1, 2)],
            [(2, 2)],
        ]
    )
    for schedule in _clock_cycles(m, n):
        assert sorted(schedule) == sorted(next(gold_schedule))


@pytest.mark.a5_2_1
def test_split_module_0():
    # Use MPS for the first device and CPU for the second to emulate a multi-device setup
    model = nn.Sequential(
        nn.Conv2d(10, 20, 5).to("mps"),
        nn.Conv2d(20, 64, 5).to("mps"),
        nn.Conv2d(64, 128, 5).to("cpu"),
    )
    partitions, devices = _split_module(model)
    assert len(partitions) == 2
    assert len(devices) == 2
    assert len(partitions[0]) == 2
    assert next(partitions[0].parameters()).device == devices[0]
    assert next(partitions[1].parameters()).device == devices[1]


@pytest.mark.a5_2_1
def test_split_module_1():
    model = nn.Sequential(
        nn.Conv2d(10, 20, 5).to("mps:0"),
        WithDevice(nn.Dropout(0.5), "mps:0"),
        nn.Conv2d(20, 64, 5).to("cpu"),
    )
    partitions, devices = _split_module(model)
    assert len(partitions) == 2
    assert len(devices) == 2
    assert next(partitions[0].parameters()).device == devices[0]
    assert next(partitions[1].parameters()).device == devices[1]


@pytest.mark.a5_2_2
@pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
@pytest.mark.parametrize("split_size", [1, 2, 4, 8, 16])
def test_forward_0(batch_size, split_size):
    # Build model initially on MPS
    model = nn.Sequential(
        nn.Linear(3, 4).to("mps"),
        WithDevice(nn.Sigmoid(), "mps"),
        nn.Linear(4, 5).to("mps"),
        WithDevice(nn.Sigmoid(), "mps"),
    )

    x = torch.randn(batch_size, 3).to("mps")
    y0 = model(x).to("cpu")

    # move the last two layer to another device (use CPU as the second device)
    model[-2] = model[-2].to("cpu")
    model[-1] = WithDevice(nn.Sigmoid(), "cpu")
    pipe = Pipe(model, split_size=split_size)
    y1 = pipe(x).to("cpu")
    assert torch.allclose(y0, y1)
