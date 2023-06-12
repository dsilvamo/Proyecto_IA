# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:38:09 2023

@author: dsilv
"""
#%% Ubicaciones 
###############################
#Ubicacion
###############################
import os
os.getcwd()
os.chdir("C:/Users/dsilv/OneDrive/Escritorio/PONG")
os.getcwd()
#%% instalaciones 
pip install stable-baselines3[extra]
pip install numpy
#%%librerias 
import numpy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C

#%% DIrecciones 
models_dir="models/A2C"
logdir="logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
#%% Implementacion A2C
vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=0)

#Modelo
model = A2C("CnnPolicy", vec_env, verbose=1,tensorboard_log=logdir)


#Entrenamiento
TIMESTEPS=100000

for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False,tb_log_name="A2C")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
#tensorboard --logdir logs
    
#%% Carga del modelo
model=A2C.load(f"{models_dir}/{2400000}",env=vec_env)
#%% Visualizaciones

while not done:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

env.close()