# monitors/ — manual registry of discovered conservation-law monitors
#
# Evolution rule (2026-03-13):
#   New monitors start as individual files here (one file = one discovery).
#   After 3-5 discoveries, common patterns get factored into registry.py.
#   Until then: manual addition = maximum flexibility, stays grounded in real signals.
#
# To add a new monitor:
#   1. Create monitors/your_quantity.py with a compute_* function
#   2. Import it here
#   3. Add one line to MONITORS dict
#   4. Add one line to your problem.yaml under monitors:
#
# Current monitors:
#   margin_sign                    — oracle log-prob margin (always required)
#   sum_pairwise_distances_variance — r12+r13+r23 near-conservation (figure-8 Z3)

from .sum_pairwise_distances import compute_vectorised as sum_pairwise_distances_variance

MONITORS = {
    # Always required — language model oracle filter
    "margin_sign": None,  # implemented in oracle_wrapper.py

    # Formal checker monitors — numerical integration
    "sum_pairwise_distances_variance": sum_pairwise_distances_variance,

    # Stubs for future monitors (add imports above when files exist):
    # "e2_symmetric_poly_variance": None,   # r12*r13 + r12*r23 + r13*r23
    # "rms_pairwise_variance": None,        # sqrt((r12^2+r13^2+r23^2)/3)
    # "energy_violation": None,             # |E(t) - E(0)| / |E(0)|
    # "rho_expression_unlock": None,        # expression gap from rho-unlock
}


def get_monitor(name: str):
    """Return the monitor function by name, or None if not yet implemented."""
    return MONITORS.get(name)


def list_monitors():
    return list(MONITORS.keys())
