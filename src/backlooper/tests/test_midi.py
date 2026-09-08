import sys
import types
from unittest import TestCase


fake_mido = types.ModuleType('mido')
fake_mido.Message = object
fake_mido.ports = types.SimpleNamespace(BaseInput=object)
sys.modules.setdefault('mido', fake_mido)

from backlooper.midi import _bars_from_fader


class TestBarsFromFader(TestCase):
    def test_fader_selects_all_supported_bar_lengths(self):
        values = (0, 16, 32, 48, 64, 80, 96, 112)
        expected = (1, 2, 4, 8, 12, 16, 24, 32)

        self.assertEqual([_bars_from_fader(value) for value in values], list(expected))

    def test_maximum_fader_value_selects_longest_loop(self):
        self.assertEqual(_bars_from_fader(127), 32)
