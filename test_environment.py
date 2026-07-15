import unittest
from unittest.mock import Mock, patch

from environment import criar_ambiente


class EnvironmentTest(unittest.TestCase):
    @patch("environment.pong_v3.env")
    def test_configura_probabilidade_de_repetir_acao(self, criar_env):
        env = Mock()
        criar_env.return_value = env

        resultado = criar_ambiente(probabilidade_acao_repetida=0.25)

        self.assertIs(resultado, env)
        env.unwrapped.ale.setFloat.assert_called_once_with(
            b"repeat_action_probability", 0.25
        )


if __name__ == "__main__":
    unittest.main()
