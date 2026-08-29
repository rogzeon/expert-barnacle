"""
Tests for pinn.model: shape checks on FCN/ConstrainedFCN, and  verification of
ConstrainedFCN's ansatz.
"""

import torch

from pinn.model import FCN, ConstrainedFCN


def test_fcn_forward_shape():
    model = FCN(input_dim=2, output_dim=1, hidden_dim=16, num_layers=3)
    x = torch.randn(32, 2)
    out = model(x)
    assert out.shape == (32, 1)


def test_fcn_metadata_matches_constructor_args():
    model = FCN(input_dim=2, output_dim=1, hidden_dim=16, num_layers=3)
    assert model.metadata == {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dim": 16,
        "num_layers": 3,
    }


def test_fcn_num_layers_controls_depth():
    shallow = FCN(input_dim=2, output_dim=1, hidden_dim=8, num_layers=1)
    deep = FCN(input_dim=2, output_dim=1, hidden_dim=8, num_layers=4)
    # num_layers-1 hidden Sequential blocks live in .fch
    assert len(shallow.fch) == 0
    assert len(deep.fch) == 3


def test_constrained_fcn_forward_shape():
    model = ConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=16,
        num_layers=3,
        strike=100.0,
        t_max=1.0,
        kink_width=0.05,
    )
    xt = torch.randn(10, 2)
    out = model(xt)
    assert out.shape == (10, 1)


def test_constrained_fcn_payoff_is_call_payoff():
    model = ConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=8,
        num_layers=2,
        strike=100.0,
        t_max=1.0,
        kink_width=0.05,
    )
    x = torch.tensor([[80.0], [100.0], [120.0]])
    torch.testing.assert_close(model.payoff(x), torch.tensor([[0.0], [0.0], [20.0]]))


def test_constrained_fcn_weight_is_one_at_t_max():
    model = ConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=8,
        num_layers=2,
        strike=100.0,
        t_max=1.0,
        kink_width=0.05,
    )
    w = model.weight(torch.tensor([[1.0]]))
    torch.testing.assert_close(w, torch.tensor([[1.0]]))


def test_constrained_fcn_hardcodes_payoff_at_t_max():
    """
    The whole point of the ansatz is u(x, t_max) = payoff(x), regardless of
    what the underlying base_net predicts. Verify that algebraic guarantee
    holds exactly (to float precision) rather than just checking shapes.
    """
    model = ConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=16,
        num_layers=3,
        strike=100.0,
        t_max=1.0,
        kink_width=0.05,
    )
    model.eval()
    x = torch.tensor([70.0, 100.0, 130.0, 200.0]).unsqueeze(1)
    t = torch.full_like(x, model.t_max)
    xt = torch.cat([x, t], dim=1)

    with torch.no_grad():
        out = model(xt)
        expected = model.payoff(x)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_constrained_fcn_weight_decays_away_from_t_max():
    model = ConstrainedFCN(
        input_dim=2,
        output_dim=1,
        hidden_dim=8,
        num_layers=2,
        strike=100.0,
        t_max=1.0,
        kink_width=0.1,
    )
    w_near = model.weight(torch.tensor([[0.95]]))
    w_far = model.weight(torch.tensor([[0.0]]))
    assert w_near.item() > w_far.item()
    assert 0.0 <= w_far.item() < w_near.item() <= 1.0
