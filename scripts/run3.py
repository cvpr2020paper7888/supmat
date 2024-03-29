
import random, sys, os, glob, yaml, time, shlex
import argparse, subprocess, shutil
from fire import Fire


def experiment_id(config):
    i = 1
    while subprocess.call(f"gsutil stat gs://taskonomy-code/{config}_{i}/utils.py".split()) == 0:
        i += 1
    return config + "_" + str(i)

def upload(exp_id):
    os.system(f"echo {exp_id}, 0, mount > scripts/jobinfo.txt")
    subprocess.run(["rsync", "-av", "--progress", ".", "checkpoints/" + exp_id, "--exclude",
        "checkpoints", "--exclude", ".git", "--exclude", "data/snapshots", 
        "--exclude", "data/results", "--exclude", "local", "--exclude", "data/alllinks.txt", 
        "--exclude", "__pycache__", "--exclude", "experiments/__pycache__"])
    subprocess.run(f"gsutil -m cp -r checkpoints/{exp_id} gs://taskonomy-code".split())

def delete(env_prefix):
    import visdom
    link = visdom.Visdom(server="http://35.229.22.191", port=7000, env=env)
    envs = link.get_env_list()
    
    for env in envs:
        if env[0:len(env_prefix)] == env_prefix:
            link.delete_env(env)

def save():
    import visdom
    link = visdom.Visdom(server="http://35.229.22.191", port=7000)
    envs = link.get_env_list()
    link.save(envs)

def run(cmd, instance="cloud1", zone="us-west1-b", config="job", shutdown=False, debug=False):
    exp_id = experiment_id(config)
    print ("Experiment ID: ", exp_id)

    # create a file called command.txt
    os.system(f"echo {cmd} > command.txt")
    upload(exp_id)

    subprocess.run(f"gcloud compute instances start {instance} --zone {zone}".split())
    subprocess.run(f"gcloud compute config-ssh".split())

    cmd = shlex.split(cmd)
    if cmd[0] == "python":
        cmd[0] = "/home/shared/anaconda3/bin/python"
        cmd.insert(0, "sudo")
    cmd = " ".join(cmd)

    with open('../../.sshrc', 'w') as outfile:
        print (f"gsutil -m cp -r gs://taskonomy-code/{exp_id}/* .", file=outfile)
        print (f"""sudo /home/shared/anaconda3/bin/python -m scripts.run2 run --config "{config}" --experiment-id "{exp_id}" --shutdown {shutdown} "{cmd}" """, file=outfile)
    
    print (cmd)
    subprocess.run(["sshrc", f"{instance}.{zone}.chaos-theory-201106"])





if __name__ == "__main__":
    Fire()
