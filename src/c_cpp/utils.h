#ifndef JSON_UTIL_H
#define JSON_UTIL_H

/*
funcoes independentes da logica do simulador
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

void concatenarStrings(
    char **str,
    const char* const bufferStr1,
    const char* const bufferStr2
);

#endif