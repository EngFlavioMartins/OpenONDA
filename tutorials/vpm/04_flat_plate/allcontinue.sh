#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.tutorial_runner . setup --mode moving --angle -10
python -m openonda.tutorial_runner . setup --mode moving --angle -5
python -m openonda.tutorial_runner . setup --mode moving --angle -2
python -m openonda.tutorial_runner . setup --mode moving --angle 0
python -m openonda.tutorial_runner . setup --mode moving --angle 2
python -m openonda.tutorial_runner . setup --mode moving --angle 5
python -m openonda.tutorial_runner . setup --mode moving --angle 8
python -m openonda.tutorial_runner . setup --mode moving --angle 10
python -m openonda.tutorial_runner . setup --mode moving --angle 12
python -m openonda.tutorial_runner . setup --mode moving --angle 15
python -m openonda.tutorial_runner . setup --mode static --angle -10
python -m openonda.tutorial_runner . setup --mode static --angle -5
python -m openonda.tutorial_runner . setup --mode static --angle -2
python -m openonda.tutorial_runner . setup --mode static --angle 0
python -m openonda.tutorial_runner . setup --mode static --angle 2
python -m openonda.tutorial_runner . setup --mode static --angle 5
python -m openonda.tutorial_runner . setup --mode static --angle 8
python -m openonda.tutorial_runner . setup --mode static --angle 10
python -m openonda.tutorial_runner . setup --mode static --angle 12
python -m openonda.tutorial_runner . setup --mode static --angle 15
