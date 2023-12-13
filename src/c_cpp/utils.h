#ifndef PROJECT_UTILS_H
#define PROJECT_UTIL_H

/*
funcoes independentes da logica do simulador
*/

#include "cJSON.h"

// =============================================================================
# define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, 0); }

inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    int abort
) {
    if (code != cudaSuccess) {
        fprintf(
            stderr,
            "GPUassert: %s %s %d\n",
            cudaGetErrorString(code),
            file,
            line
        );
        if (abort) {
            exit(code);
        }
    }
}

cJSON *carregarJSON(
    const char *jsonFilePath
);

int descobrirTamanhoMatriz(
    size_t *bufferNumeroLinhas,
    size_t *bufferNumeroColunas,
    cJSON *const matriz
);

int copiarMatrizIntJsonParaArray(
    int *bufferArray,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
);

int copiarMatrizFloatJsonParaArray(
    float *bufferArray,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
);

void copiarColunasMatrizJsonParaArrays_int(
    int **buffersArrays,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
);

void copiarColunasMatrizJsonParaArrays_float(
    float **buffersArrays,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
);

void concatenarStrings(
    char **str,
    const char* const bufferStr1,
    const char* const bufferStr2
);

void dateTimeAsString(
    char **string
);
#endif