#! /usr/bin/bash
nvcc -o ./output/simulador.bin ./src/c_cpp/main.cu ./src/c_cpp/cJSON.c ./src/c_cpp/utils.cu ./src/c_cpp/simulador_gpu_1.cu ./src/c_cpp/simulador_gpu_2.cu -lcublas -lcusolver
