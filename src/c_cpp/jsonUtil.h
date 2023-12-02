#ifndef JSON_UTIL_H
#define JSON_UTIL_H

/*
funçoes sobre json que são independentes do simulador
*/

#include "cJSON.h"

// =============================================================================
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

#endif