"""
CPU offloading of layer data (``offload_device="cpu"``) on a CUDA simulation.

Checks that offloading leaves the results unchanged and that it actually lowers
the peak GPU memory. Only meaningful on a GPU, so it is skipped with
``--device=cpu`` and runs in the GPU CI.

The structure (20 patterned layers at Fourier order 10, i.e. 882x882 complex64
matrices) is large enough that the per-layer data dominates the peak memory:
on an RTX 5070 Ti the peak drops from about 2.6 GiB to 1.2 GiB.
"""

import pytest
import torch

import solwa

LAMBDA = 1.3
PERIOD = 1.0
ORDER = [10, 10]
N_LAYERS = 20
DTYPE = torch.complex64
# Offloading must save at least this fraction of the peak GPU memory (measured: ~56 %).
MIN_SAVING = 0.4


def _patterns():
    gen = torch.Generator().manual_seed(0)
    return [
        1.0 + 3.0 * (torch.rand(64, 64, generator=gen) > 0.5).float()
        for _ in range(N_LAYERS)
    ]


def _run(device, offload_device):
    """Build and solve the stack; return results and the peak allocated GPU memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    sim = solwa.rcwa(
        freq=1 / LAMBDA,
        order=ORDER,
        L=[PERIOD, PERIOD],
        dtype=DTYPE,
        device=device,
        offload_device=offload_device,
    )
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    for eps in _patterns():
        sim.add_layer(thickness=0.1, eps=eps.to(device))
    sim.solve_global_smatrix()

    orders = [[0, 0], [1, 0], [0, 1], [-1, -1]]
    results = {
        (port, pol): sim.S_parameters(
            orders=orders, direction="forward", port=port, polarization=pol
        )
        for port in ("transmission", "reflection")
        for pol in ("xx", "yy", "xy")
    }

    # Fields inside an inner layer read back the offloaded eigenvectors and C.
    sim.source_planewave(amplitude=[1.0, 0.0], direction="forward")
    axis = torch.linspace(0, PERIOD, 16, device=device)
    results["field"] = torch.stack(
        sim.field_xy(N_LAYERS // 2, axis, axis, z_prop=0.05)[0]
    )

    if offload_device is not None:
        assert sim.layer_S11[0].device.type == "cpu"
        assert sim.E_eigvec[N_LAYERS // 2].device.type == "cpu"

    results = {k: v.detach().cpu() for k, v in results.items()}
    peak = torch.cuda.max_memory_allocated(device) - base
    del sim
    return results, peak


def test_cpu_offloading_matches_and_saves_vram(device):
    if device.type != "cuda":
        pytest.skip("offloading only saves GPU memory; run with --device=cuda")

    ref, peak_ref = _run(device, None)
    off, peak_off = _run(device, "cpu")

    for key in ref:
        assert torch.isfinite(off[key]).all(), key
        torch.testing.assert_close(off[key], ref[key], msg=f"mismatch in {key}")

    saving = 1 - peak_off / peak_ref
    print(
        f"peak GPU memory: {peak_ref / 2**20:.0f} MiB without offloading, "
        f"{peak_off / 2**20:.0f} MiB with offloading ({saving:.0%} saved)"
    )
    assert (
        saving >= MIN_SAVING
    ), f"offloading saved only {saving:.0%} of peak GPU memory"
