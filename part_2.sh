#!/bin/bash

python gen.py 

mpiexec -n 4 python main.py &