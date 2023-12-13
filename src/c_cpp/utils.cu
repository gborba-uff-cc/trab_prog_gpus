#include <stdio.h>
#include <time.h>

#include "utils.h"

// =============================================================================
#define TIPO_CARACTERE_ARQUIVO char
#define MAXIMO_BYTES_ARQUIVO_JSON 1000001
#define MAXIMO_CARACTERES_BUFFER_ARQUIVO MAXIMO_BYTES_ARQUIVO_JSON / sizeof(TIPO_CARACTERE_ARQUIVO)

cJSON *carregarJSON(
    const char *jsonFilePath
) {
    FILE *fp;
    fp = fopen(jsonFilePath, "rb");
    if (!fp) {
        return NULL;
    }

    TIPO_CARACTERE_ARQUIVO jsonString[MAXIMO_CARACTERES_BUFFER_ARQUIVO];
    size_t elementosLidos = fread(
        jsonString,
        sizeof(char),
        MAXIMO_CARACTERES_BUFFER_ARQUIVO,
        fp
    );
    if (ferror(fp)) {
        fclose(fp);
        *jsonString = '\0';
        return NULL;
    }
    else {
        jsonString[elementosLidos] = '\0';
    }
    fclose(fp);

    cJSON *jsonStruct = cJSON_Parse(jsonString);
    if (jsonStruct == NULL) {
        const char *erro = cJSON_GetErrorPtr();
        if (erro != NULL) {
            fprintf(stderr, "Erro antes de: %s", erro);
        }
    }
    return jsonStruct;
}

int descobrirTamanhoMatriz(
    size_t *bufferNumeroLinhas,
    size_t *bufferNumeroColunas,
    cJSON *const matriz
) {
    cJSON *linha = NULL, *coluna = NULL;
    size_t l = 0, c = 0, c0 = 0;

    // numero de colunas na linha 0
    cJSON_ArrayForEach(coluna, matriz->child) {
        c0++;
    }

    // numero de linhas
    cJSON_ArrayForEach(linha, matriz) {
        c = 0;
        // numero de colunas
        cJSON_ArrayForEach(coluna, linha) {
            c++;
        }
        // verifica se todas as colunas tem mesmo tamanho
        if (c != c0) {
            return 1;
        }
        l++;
    }
    *bufferNumeroLinhas  = l;
    *bufferNumeroColunas = c0;
    return 0;
}

int copiarMatrizIntJsonParaArray(
    int *bufferArray,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
) {
    cJSON *linha = NULL, *coluna = NULL;
    size_t l = 0, c = 0, i = 0;

    cJSON_ArrayForEach(linha, matriz) {
        c = 0;
        cJSON_ArrayForEach(coluna, linha) {
            i = l*matrizNumeroColunas+c;
            bufferArray[i] = (int) cJSON_GetNumberValue(coluna);
            c++;
        }
        l++;
    }
    return 0;
}

int copiarMatrizFloatJsonParaArray(
    float *bufferArray,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
) {
    cJSON *linha = NULL, *coluna = NULL;
    size_t l = 0, c = 0, i = 0;

    cJSON_ArrayForEach(linha, matriz) {
        c = 0;
        cJSON_ArrayForEach(coluna, linha) {
            i = l*matrizNumeroColunas+c;
            bufferArray[i] = (float) cJSON_GetNumberValue(coluna);
            c++;
        }
        l++;
    }
    return 0;
}

void copiarColunasMatrizJsonParaArrays_int(
    int **buffersArrays,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
) {
    cJSON *linha = NULL, *coluna = NULL;
    size_t l = 0, c = 0;

    cJSON_ArrayForEach(linha, matriz) {
        c = 0;
        cJSON_ArrayForEach(coluna, linha) {
            buffersArrays[c][l] = (int) cJSON_GetNumberValue(coluna);
            c++;
        }
        l++;
    }
    return;
}

void copiarColunasMatrizJsonParaArrays_float(
    float **buffersArrays,
    cJSON *const matriz,
    size_t matrizNumeroLinhas,
    size_t matrizNumeroColunas
) {
    cJSON *linha = NULL, *coluna = NULL;
    size_t l = 0, c = 0;

    cJSON_ArrayForEach(linha, matriz) {
        c = 0;
        cJSON_ArrayForEach(coluna, linha) {
            buffersArrays[c][l] = (float) cJSON_GetNumberValue(coluna);
            c++;
        }
        l++;
    }
    return;
}

void concatenarStrings(char **str, const char* const bufferStr1, const char* const bufferStr2) {
    size_t tamanhoBuffer = strlen(bufferStr1) + strlen(bufferStr2) + 1;
    char* const buffer = (char *) malloc(tamanhoBuffer);

    strcpy(buffer, bufferStr1);
    strcat(buffer, bufferStr2);

    *str = buffer;
}

void dateTimeAsString(
    char **string
) {
    const size_t bytesString = sizeof(char)*20;
    char *s =(char *) malloc(bytesString);
    const char* formato = "%Y%m%d%H%M";
    time_t horarioAtual = time(NULL);

    strftime(s, bytesString, formato, localtime(&horarioAtual));
    *string = s;
    return;
}
