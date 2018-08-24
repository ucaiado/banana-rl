Usseful Notes
=============

From Daniel Barbosa

#### Launch Instance
- go to us-east-1
- create spot instance with the following parameters:
  - AMI: ami-18642967
  - type: p2.xlarge
  - security group: ssh-inbound-only
  - key pair: <your key>
#### SSH
- add SSH key (if needed): `ssh-add`
- ssh to IP

#### Install ML-agents
source activate pytorch_p36
cd ml-agents
git pull
cd ml-agents/python
pip install .

#### Launch X server
sudo /usr/bin/X :0 &
nvidia-smi
export DISPLAY=:0

#### Install unity environments
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip
unzip VisualBanana_Linux.zip

#### Test it's all working
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="VisualBanana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
info = env.reset(train_mode=True)[brain_name]
state = info.visual_observations[0]
