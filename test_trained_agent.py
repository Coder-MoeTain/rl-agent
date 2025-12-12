"""Test and evaluate trained agents."""
import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from gym_pentest.env import PentestEnv
from gpu_utils import setup_gpu

def test_agent(model_path, num_episodes=10, render=False):
    """Test a trained agent.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        num_episodes: Number of episodes to test
        render: Whether to print episode details
    """
    print("=" * 70)
    print(f"Testing Agent: {model_path}")
    print("=" * 70)
    
    # Load model
    try:
        print(f"\n[INFO] Loading model from '{model_path}.zip'...")
        model = PPO.load(model_path)
        print(f"[OK] Model loaded successfully")
        print(f"[INFO] Model device: {model.device}")
    except FileNotFoundError:
        print(f"[ERROR] Model file '{model_path}.zip' not found!")
        print("\nAvailable models:")
        for p in Path('.').glob('*.zip'):
            print(f"  - {p.stem}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None
    
    # Create environment
    print("\n[INFO] Creating test environment...")
    env = PentestEnv()
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    vulnerabilities_found = []
    endpoints_discovered = []
    
    print(f"\n[INFO] Running {num_episodes} test episodes...")
    print("-" * 70)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_vulns = 0
        episode_endpoints = set()
        
        while not (done or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Track discoveries
            if 'vulnerabilities' in info:
                episode_vulns = info['vulnerabilities']
            if 'discovered_count' in info:
                episode_endpoints.add(info['discovered_count'])
        
        # Final stats
        final_vulns = sum(len(env.attack_graph.nodes[n].get('vulns', [])) 
                         for n in env.attack_graph.nodes)
        final_endpoints = len(env.discovered)
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        vulnerabilities_found.append(final_vulns)
        endpoints_discovered.append(final_endpoints)
        
        # Print episode summary
        if render:
            print(f"\nEpisode {episode + 1}:")
            print(f"  Reward: {total_reward:7.2f}")
            print(f"  Steps: {steps:3d}")
            print(f"  Vulnerabilities found: {final_vulns}")
            print(f"  Endpoints discovered: {final_endpoints}")
        else:
            print(f"Episode {episode + 1:3d}: "
                  f"Reward: {total_reward:7.2f} | "
                  f"Steps: {steps:3d} | "
                  f"Vulns: {final_vulns} | "
                  f"Endpoints: {final_endpoints}")
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    
    print(f"\nEpisodes Tested: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Mean:   {np.mean(episode_rewards):7.2f}")
    print(f"  Std:    {np.std(episode_rewards):7.2f}")
    print(f"  Min:    {np.min(episode_rewards):7.2f}")
    print(f"  Max:    {np.max(episode_rewards):7.2f}")
    print(f"  Median: {np.median(episode_rewards):7.2f}")
    
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean:   {np.mean(episode_lengths):5.1f}")
    print(f"  Std:    {np.std(episode_lengths):5.1f}")
    print(f"  Min:    {np.min(episode_lengths):5.0f}")
    print(f"  Max:    {np.max(episode_lengths):5.0f}")
    
    print(f"\nVulnerabilities Found:")
    print(f"  Mean per episode: {np.mean(vulnerabilities_found):.1f}")
    print(f"  Total found:      {sum(vulnerabilities_found)}")
    print(f"  Episodes with vulns: {sum(1 for v in vulnerabilities_found if v > 0)}/{num_episodes}")
    
    print(f"\nEndpoints Discovered:")
    print(f"  Mean per episode: {np.mean(endpoints_discovered):.1f}")
    print(f"  Max discovered:   {max(endpoints_discovered)}")
    
    # Success rate (positive reward episodes)
    success_rate = sum(1 for r in episode_rewards if r > 0) / num_episodes * 100
    print(f"\nSuccess Rate: {success_rate:.1f}% (episodes with positive reward)")
    
    print("=" * 70)
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'vulnerabilities': vulnerabilities_found,
        'endpoints': endpoints_discovered,
        'success_rate': success_rate
    }

def compare_agents(model_paths, num_episodes=10):
    """Compare multiple trained agents.
    
    Args:
        model_paths: List of model paths to compare
        num_episodes: Number of episodes per model
    """
    print("=" * 70)
    print("Agent Comparison")
    print("=" * 70)
    
    results = {}
    
    for model_path in model_paths:
        print(f"\nTesting: {model_path}")
        result = test_agent(model_path, num_episodes, render=False)
        if result:
            results[model_path] = result
    
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("Comparison Summary")
        print("=" * 70)
        
        print(f"\n{'Model':<30} {'Mean Reward':<15} {'Success Rate':<15} {'Avg Vulns':<15}")
        print("-" * 70)
        
        for model_path, result in results.items():
            mean_reward = np.mean(result['rewards'])
            success_rate = result['success_rate']
            avg_vulns = np.mean(result['vulnerabilities'])
            
            print(f"{model_path:<30} {mean_reward:>10.2f}     {success_rate:>10.1f}%     {avg_vulns:>10.1f}")
        
        print("=" * 70)
    
    return results

def watch_agent_play(model_path, num_episodes=3):
    """Watch agent play with detailed output.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to watch
    """
    print("=" * 70)
    print(f"Watching Agent Play: {model_path}")
    print("=" * 70)
    
    # Load model
    try:
        model = PPO.load(model_path)
        print(f"[OK] Model loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load: {e}")
        return
    
    env = PentestEnv()
    
    action_names = [
        "Crawl root",
        "GET /login",
        "GET /feedback",
        "POST login",
        "XSS inject (script)",
        "GET /products",
        "GET /whoami",
        "XSS inject (img)"
    ]
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}")
        print(f"{'='*70}")
        
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step += 1
            
            print(f"\nStep {step}:")
            print(f"  Action: {action} ({action_names[action] if action < len(action_names) else 'Unknown'})")
            print(f"  Reward: {reward:+.2f}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Discovered: {info.get('discovered_count', 0)} endpoints")
            print(f"  Vulnerabilities: {info.get('vulnerabilities', 0)}")
            
            if done or truncated:
                print(f"\n  Episode ended: {'Done' if done else 'Truncated'}")
        
        print(f"\nEpisode Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Final endpoints: {len(env.discovered)}")
        print(f"  Final vulnerabilities: {sum(len(env.attack_graph.nodes[n].get('vulns', [])) for n in env.attack_graph.nodes)}")

def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained agents')
    parser.add_argument('--model', type=str, default='ppo_baseline',
                       help='Model path (without .zip)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--watch', action='store_true',
                       help='Watch agent play with detailed output')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_agents(args.compare, args.episodes)
    elif args.watch:
        watch_agent_play(args.model, args.episodes)
    else:
        test_agent(args.model, args.episodes, render=False)

if __name__ == '__main__':
    main()

