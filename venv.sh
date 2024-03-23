#!/bin/bash

PWD=`pwd`
python3.10 -m venv slamenv

activate_env(){
    . $PWD/slamenv/bin/activate
}

activate_env

pip install -r requirements.txt
