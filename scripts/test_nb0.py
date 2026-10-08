"""Tests for NB0 (00_dpo_loss_from_scratch.py) CPU implementation.

Extracts and verifies my_dpo_loss from the notebook source without running
the whole notebook.
"""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")
from lab22 import dpo_math as M
NB0_PATH = REPO / "notebooks" / "00_dpo_loss_from_scratch.py"


def _load_nb0_my_dpo_loss() -> tuple[object, str]:
    source = NB0_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(NB0_PATH))
    func_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "my_dpo_loss"),
        None,
    )
    assert func_node is not None, "my_dpo_loss function definition not found in NB0"
    func_source = ast.get_source_segment(source, func_node)
    assert func_source is not None, "failed to get source segment for my_dpo_loss"
    mod = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    code = compile(mod, filename=str(NB0_PATH), mode="exec")
    namespace: dict[str, object] = {"torch": torch}
    exec(code, namespace)
    return namespace["my_dpo_loss"], func_source


def test_my_dpo_loss_matches_reference():
    my_dpo_loss, _ = _load_nb0_my_dpo_loss()
    pc, pr = torch.tensor([-12.0, -30.0]), torch.tensor([-15.0, -28.0])
    rc, rr = torch.tensor([-13.0, -29.0]), torch.tensor([-14.0, -29.0])
    for beta in (0.05, 0.1, 0.5):
        ref_loss, _, _ = M.dpo_loss(pc, pr, rc, rr, beta=beta)
        loss = my_dpo_loss(pc, pr, rc, rr, beta=beta)
        assert torch.allclose(torch.as_tensor(loss), ref_loss, atol=1e-6)


def test_my_dpo_loss_is_log2_at_init():
    my_dpo_loss, _ = _load_nb0_my_dpo_loss()
    same = torch.tensor([-20.0, -35.0])
    loss = my_dpo_loss(same, same - 3.0, same, same - 3.0, beta=0.1)
    assert float(torch.as_tensor(loss).item()) == pytest.approx(math.log(2), abs=1e-6)
    default_loss = my_dpo_loss(same, same - 3.0, same, same - 3.0)
    assert float(torch.as_tensor(default_loss).item()) == pytest.approx(math.log(2), abs=1e-6)


def test_my_dpo_loss_has_no_todo():
    _, func_source = _load_nb0_my_dpo_loss()
    assert "TODO" not in func_source
    assert "return None" not in func_source
