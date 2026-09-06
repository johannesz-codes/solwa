"""Select the execution device explicitly; CUDA CI must never silently skip."""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", choices=("cpu", "cuda"), default="cpu")


def pytest_sessionstart(session):
    if session.config.getoption("--device") == "cuda":
        if not torch.cuda.is_available():
            raise pytest.UsageError("--device=cuda requires a working CUDA GPU")
        # Exercise initialization and an actual kernel, not only device discovery.
        assert (torch.ones(2, device="cuda") + 1).sum().item() == 4


@pytest.fixture(scope="session")
def device(request):
    return torch.device(request.config.getoption("--device"))
