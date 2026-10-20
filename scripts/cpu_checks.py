"""Dependency-free teaching checks. These are NOT model evaluation results."""
import math


def masked_nll(log_probs, supervised):
    if len(log_probs) != len(supervised):
        raise ValueError("Mismatched lengths")
    selected = [p for p, keep in zip(log_probs, supervised) if keep]
    if not selected:
        raise ValueError("No supervised targets")
    return -sum(selected) / len(selected)


def dpo_loss(pc, pr, rc, rr, beta=0.1):
    if beta <= 0:
        raise ValueError("beta must be positive")
    z = beta * ((pc - rc) - (pr - rr))
    return max(0.0, -z) + math.log1p(math.exp(-abs(z)))


def wilson(k, n, z=1.959963984540054):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Require 0 <= k <= n and n > 0")
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return max(0.0, center-radius), min(1.0, center+radius)


def authorize(action, recipients, write_granted=False):
    if action.get("tool") == "read_document":
        return True
    if action.get("tool") == "send_message":
        return write_granted is True and action.get("recipient") in recipients
    return False


def main():
    assert masked_nll([-8., -2., -4.], [False, True, True]) == 3.
    assert math.isclose(dpo_loss(-3., -4., -3., -4.), math.log(2))
    assert dpo_loss(-2., -4., -3., -3.) < dpo_loss(-4., -2., -3., -3.)
    assert math.isfinite(dpo_loss(-10000., 0., 0., 0.))
    lo, hi = wilson(0, 100)
    assert lo < 1e-12 and 0.03 < hi < 0.04
    assert not authorize({"tool": "send_message", "recipient": "other@example.invalid"},
                         {"allowed@example.invalid"}, True)
    assert authorize({"tool": "send_message", "recipient": "allowed@example.invalid"},
                     {"allowed@example.invalid"}, True)
    print("PASS: SFT mask, DPO math, Wilson interval, simulated authorization.")
    print("Teaching checks only. No model training, benchmark, network, or real tool action ran.")


if __name__ == "__main__":
    main()
