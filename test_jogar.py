import unittest
from unittest.mock import patch

from config import CAMINHO_MELHOR_CHECKPOINT
from jogar import resolver_modelo


class JogarTest(unittest.TestCase):
    @patch("jogar.os.path.exists", return_value=True)
    def test_usa_melhor_checkpoint_por_padrao(self, _):
        self.assertEqual(resolver_modelo(None), CAMINHO_MELHOR_CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
