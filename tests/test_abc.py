import unittest
from modules.abc import add

class TestAddFuction(unittest.TestCase):
    def test_add_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

if __name__ == '__main__':
    unittest.main()