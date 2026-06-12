import torch

from solwa.materials import Material


def _write_nk_file(path):
    path.write_text(
        "\n".join(
            [
                "400 1.50 0.05",
                "500 1.60 0.10",
                "600 1.70 0.15",
                "700 1.80 0.20",
            ]
        )
    )


def test_material_default_mode_uses_file_k(tmp_path):
    nk_file = tmp_path / "nk.txt"
    _write_nk_file(nk_file)

    material = Material(str(nk_file))
    nk = material.apply(torch.tensor(550.0))

    assert torch.imag(nk).item() > 0.0


def test_material_lossless_mode_zeros_k(tmp_path):
    nk_file = tmp_path / "nk.txt"
    _write_nk_file(nk_file)

    material = Material(str(nk_file), lossless=True)
    nk = material.apply(torch.tensor(550.0))

    assert torch.imag(nk).item() == 0.0


def test_material_lossless_keeps_real_part(tmp_path):
    nk_file = tmp_path / "nk.txt"
    _write_nk_file(nk_file)

    material = Material(str(nk_file))
    material_lossless = Material(str(nk_file), lossless=True)
    wavelength = torch.tensor(550.0)

    n_complex = material.apply(wavelength)
    n_lossless = material_lossless.apply(wavelength)

    assert torch.real(n_complex).item() == torch.real(n_lossless).item()
