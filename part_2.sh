#!/bin/bash

python3 gen.py 

mpiexec -n 4 python3 main.py 