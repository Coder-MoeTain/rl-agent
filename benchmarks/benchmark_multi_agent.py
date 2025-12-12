\
from agents.train_multi_agent_per_is import train as train_multi
from gym_pentest.env import PentestEnv

if __name__ == '__main__':
    print('Training multi-agent (short)...')
    # train_multi()  # Uncomment to run full multi-agent training
    env = PentestEnv()
    print('Multi-agent training placeholder — run agents/train_multi_agent_per_is.py to train full system.')
