import unittest
import numpy as np
from genetic_agent import decodificar_cromossomo, AgenteGenetico

class TestGeneticAgent(unittest.TestCase):
    def test_decodificar_cromossomo(self):
        cromossomo = [0] * 12288
        pesos = decodificar_cromossomo(cromossomo)
        self.assertEqual(pesos.shape, (128, 6))
        # Zeroes represent -32768, so mapped to -1.0
        self.assertTrue(np.all(pesos == -1.0))
        
        # Test max value
        cromossomo_max = [1] * 12288
        pesos_max = decodificar_cromossomo(cromossomo_max)
        self.assertTrue(np.all(pesos_max == 32767 / 32768.0))

    def test_escolher_acao(self):
        cromossomo = [1] * 12288
        agente = AgenteGenetico(cromossomo)
        observacao = np.zeros(128, dtype=np.uint8)
        observacao[0] = 255
        acao = agente.escolher_acao(observacao)
        self.assertTrue(0 <= acao <= 5)

if __name__ == '__main__':
    unittest.main()
