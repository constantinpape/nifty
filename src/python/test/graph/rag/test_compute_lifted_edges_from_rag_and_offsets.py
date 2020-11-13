import unittest


import numpy as np
import nifty.graph.rag as nrag


# Just a quick and dirty test to check that the functions can be called
class TestAccumulate(unittest.TestCase):
    shape_2d = 2 * (128,)
    shape_3d = 3 * (64,)

    def test_lifted_edges_2d(self):
        offsets = [
            [1, 2], # One down, two right
            [2, 0]  # Two down
        ]
        labels = np.array([
            [0, 1, 2],
            [0, 1, 1],
            [0, 1, 3]
        ], dtype='uint32')
        rag = nrag.gridRag(labels, numberOfLabels=4)
        lifted_edges = nrag.computeLiftedEdgesFromRagAndOffsets(rag, offsets)
        self.assertTrue(len(lifted_edges) == 2)
        for lifted_e in lifted_edges:
            self.assertTrue(3 in lifted_e)
            self.assertTrue(2 in lifted_e or 0 in lifted_e)



    def test_accumulate_mean_and_length_3d(self):
        labels = np.random.randint(0, 100, size=self.shape_3d, dtype='uint32')
        rag = nrag.gridRag(labels, numberOfLabels=100)
        offsets = [
            [2, 20, -7],
            [0, 0,  -5]
        ]

        lifted_edges = nrag.computeLiftedEdgesFromRagAndOffsets(rag, offsets)
        # TODO: actually test something



if __name__ == '__main__':
    unittest.main()
