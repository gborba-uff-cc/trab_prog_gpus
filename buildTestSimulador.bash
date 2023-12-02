#! /usr/bin/bash
nvcc -o ./output/simuladorTeste.bin ./src/test/test.cu ./src/c_cpp/utils.cu ./src/c_cpp/simulador_gpu_2.cu ./src/c_cpp/cJSON.c -lcublas -lcusolver

