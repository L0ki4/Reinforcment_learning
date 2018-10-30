MACHINE = 'local'

BATCH_SIZE = 24
LR = 0.1         # learning rate
EPSILON = 0.9            # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 300   # target update frequency
MEMORY_CAPACITY = 16*24

test_token = 'df133af2-dccb-4f06-b31f-bf99c02b1ba9'
train_token = 'a7bf92fc-2bd6-4ab6-9180-9f403f8d490b'

MODE = 'train'
if MODE == 'train':
    TOKEN = train_token
else:
    TOKEN = test_token

SAVE_MODELS = True
SAVE_GRAPHS = True

N_ACTIONS = 20
N_STATES = 38
