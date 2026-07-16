import numpy as np

def decodificar_cromossomo(cromossomo):
    """
    Decodifica um cromossomo de 12288 bits em uma matriz de pesos (128, 6).
    Cada peso usa 16 bits (ponto fixo).
    """
    cromossomo_array = np.array(cromossomo, dtype=np.uint8)
    
    # Redimensiona para (768, 16)
    bits_por_peso = cromossomo_array.reshape(768, 16)
    
    # Potências de 2: [2^15, 2^14, ..., 2^0]
    potencias = 1 << np.arange(15, -1, -1)
    
    # Multiplica e soma para obter inteiros entre 0 e 65535
    valores_int = bits_por_peso.dot(potencias)
    
    # Mapeia de [0, 65535] para [-1.0, 1.0)
    pesos_float = (valores_int - 32768) / 32768.0
    
    return pesos_float.reshape(128, 6)

class AgenteGenetico:
    def __init__(self, cromossomo):
        self.pesos = decodificar_cromossomo(cromossomo)
        
    def escolher_acao(self, observacao_ram):
        """
        Recebe a observacao de RAM (128 bytes), divide por 255.0 (conforme issue #7),
        multiplica pela matriz de pesos e retorna a ação de maior pontuação.
        """
        # A observação pode vir como numpy array ou lista de int/uint8
        obs_float = np.array(observacao_ram, dtype=np.float32) / 255.0
        
        # Produto escalar (1, 128) x (128, 6) = (1, 6)
        scores = np.dot(obs_float, self.pesos)
        
        return int(np.argmax(scores))
