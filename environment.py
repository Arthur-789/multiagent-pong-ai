from pettingzoo.atari import pong_v3

from config import OBS_TYPE, MAX_CYCLES

def criar_ambiente(render_mode=None, probabilidade_acao_repetida=0.0):
    env = pong_v3.env(
        obs_type=OBS_TYPE,
        max_cycles=MAX_CYCLES,
        render_mode=render_mode,
    )
    env.unwrapped.ale.setFloat(
        b"repeat_action_probability", probabilidade_acao_repetida
    )
    return env
