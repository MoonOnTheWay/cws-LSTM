#!/bin/bash
#THEANO_FLAGS='floatX=float32,device=gpu,optimizer=None' python baseline.py
THEANO_FLAGS='floatX=float32,device=gpu' python -u baseline.py
