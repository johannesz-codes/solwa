import pytest
import torch

from solwa.materials import Material


def test_material_lossless_zeros_k_and_keeps_n(tmp_path, device):
    if device.type != "cpu":
        pytest.skip("material tables are device-independent; CPU only")

    nk_file = tmp_path / "nk.txt"
    nk_file.write_text("400 1.50 0.05\n500 1.60 0.10\n600 1.70 0.15\n700 1.80 0.20\n")
    wavelength = torch.tensor(550.0)

    lossy = Material(str(nk_file)).apply(wavelength)
    lossless = Material(str(nk_file), lossless=True).apply(wavelength)

    assert lossy.imag.item() > 0.0
    assert abs(lossless.imag.item()) < 1e-9
    assert abs(lossless.real.item() - lossy.real.item()) < 1e-12
