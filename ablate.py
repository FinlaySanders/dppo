if __name__ == "__main__":
    import argparse
    from dppo import main
    import subprocess

    parser = argparse.ArgumentParser(description="Run DPPO on a specified environment.")
    parser.add_argument("--env_id", type=str, default="CartPole-v1", help="Environment ID")
    args = parser.parse_args()

    for i in range(100):
        #subprocess.run(["python", "ppo.py", "--track", "--wandb-project-name", "bin2", "--env_id", args.env_id, "--seed", str(i)])
        subprocess.run(["python", "dppo.py", "--track", "--wandb-project-name", "cartpole", "--env_id", args.env_id, "--seed", str(i)])