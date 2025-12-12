\
import unittest
from utils.prioritized_replay import PrioritizedReplay

class TestPER(unittest.TestCase):
    def test_beta_anneal(self):
        per = PrioritizedReplay(capacity=16, alpha=0.6, beta_start=0.3, beta_frames=100)
        b1 = per._beta_by_frame()
        per.frame = 50
        b2 = per._beta_by_frame()
        per.frame = 100
        b3 = per._beta_by_frame()
        self.assertTrue(b1 <= b2 <= b3)
if __name__ == '__main__':
    unittest.main()
