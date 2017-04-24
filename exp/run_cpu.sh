#!/bin/bash
#THEANO_FLAGS='floatX=float32,device=gpu,optimizer=None' python baseline.py
THEANO_FLAGS='floatX=float32,device=cpu' python -u baseline.py
