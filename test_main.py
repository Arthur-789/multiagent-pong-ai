import unittest
from unittest.mock import patch

import main


class MainTest(unittest.TestCase):
    @patch("main.avaliar")
    def test_encaminha_modelo_explicito_para_avaliacao(self, avaliar):
        with patch("sys.argv", ["main.py", "avaliar", "modelo.npz"]):
            main.main()

        avaliar.assert_called_once_with(modelo="modelo.npz", render=False)


if __name__ == "__main__":
    unittest.main()
