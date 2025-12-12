"""Quick example of testing a trained agent."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_trained_agent import test_agent, watch_agent_play

def main():
    """Quick test example."""
    print("=" * 70)
    print("Quick Agent Testing Example")
    print("=" * 70)
    
    # Try to test default model
    model_name = "ppo_baseline"
    
    print(f"\nAttempting to test model: {model_name}")
    print("(If model doesn't exist, train first with: python agents/train_single_agent.py)\n")
    
    try:
        # Basic test
        results = test_agent(model_name, num_episodes=5, render=False)
        
        if results:
            print("\n" + "=" * 70)
            print("Test completed successfully!")
            print("=" * 70)
            print("\nTo watch agent play step-by-step:")
            print(f"  python test_trained_agent.py --watch --model {model_name}")
            print("\nTo test with more episodes:")
            print(f"  python test_trained_agent.py --model {model_name} --episodes 20")
    except Exception as e:
        print(f"\n[INFO] Model not found or error: {e}")
        print("\nTo create a model, first train:")
        print("  python agents/train_single_agent.py")
        print("  python train_sb3_per.py")
        print("\nThen test with:")
        print("  python test_trained_agent.py")

if __name__ == '__main__':
    main()

