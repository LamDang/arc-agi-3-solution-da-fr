import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.events import event_row


class EventTests(unittest.TestCase):
    def test_finished_result_has_existing_elapsed_time(self):
        result = {'elapsed_seconds': 639.2, 'loss': 0.625, 'event': 'payload'}
        row = event_row('finished', 640., result)
        self.assertEqual(row, {'event': 'finished', 'elapsed_seconds': 640., 'loss': 0.625})
        self.assertEqual(result['elapsed_seconds'], 639.2)
