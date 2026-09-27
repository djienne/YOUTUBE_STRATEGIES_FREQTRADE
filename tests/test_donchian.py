"""Run with a Freqtrade environment: python -m unittest discover -s tests."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

STRATEGY = Path(__file__).resolve().parents[1] / 'Strategies' / 'Donchian' / 'user_data' / 'strategies' / 'donchian.py'
spec = importlib.util.spec_from_file_location('donchian', STRATEGY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DonchianCheck(unittest.TestCase):
    def test_bands_exclude_current_close(self):
        data = pd.DataFrame({'close': [10., 12., 11., 9., 13.]})
        module.donchian_breakout(data, 3)
        np.testing.assert_allclose(data.upper, [np.nan, 10, 12, 12, 11])
        np.testing.assert_allclose(data.lower, [np.nan, 10, 10, 11, 9])
        np.testing.assert_allclose(data.signal, [np.nan, 1, 1, -1, 1])

    def test_opposite_leg_loss_gate(self):
        data = pd.DataFrame({'close': [10., 12., 14., 11., 15., 10.]})
        signals = np.array([-1., 1., 1., -1., 1., -1.])
        # Short 10->12 loses; long 12->11 loses; short 11->15 loses.
        np.testing.assert_array_equal(
            module.last_trade_adj_signal(data, signals), [0, 1, 1, -1, 1, -1])

    def test_parameters_and_prefix_causality(self):
        params = json.loads(STRATEGY.with_suffix('.json').read_text())
        self.assertEqual(params['params']['buy']['lookback'], 185)
        strategy = module.donchian({})
        strategy.lookback.value = 185
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=800, freq='h', tz='UTC'),
            'close': 100 + 10 * np.sin(np.arange(800) / 23),
        })
        full = strategy.populate_indicators(data.copy(), {})
        prefix = strategy.populate_indicators(data.iloc[:500].copy(), {})
        pd.testing.assert_frame_equal(full.iloc[:500], prefix)


if __name__ == '__main__':
    unittest.main()
