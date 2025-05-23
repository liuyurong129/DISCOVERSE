import os

DISCOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.getenv('DISCOVERSE_ASSETS_DIR'):
    DISCOVERSE_ASSETS_DIR = os.getenv('DISCOVERSE_ASSETS_DIR')
    print(f'>>> get env "DISCOVERSE_ASSETS_DIR": {DISCOVERSE_ASSETS_DIR}')
else:
    DISCOVERSE_ASSETS_DIR = os.path.join(DISCOVERSE_ROOT_DIR, 'models')
