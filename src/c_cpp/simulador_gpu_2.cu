#include <stdio.h>
#include <stdlib.h>

#include "cJSON.h"

#define TIPO_CARACTERE_ARQUIVO char
#define MAXIMO_BYTES_ARQUIVO_JSON 1000001
#define MAXIMO_CARACTERES_BUFFER_ARQUIVO MAXIMO_BYTES_ARQUIVO_JSON / sizeof(TIPO_CARACTERE_ARQUIVO)


cJSON *carregarJSON(
    const char *jsonFilePath
);
int carregarParametros(
    float *h,
    float *k,
    int   **posicoesGrade,
    int   **conexoes,
    float **condicoesContorno,
    const cJSON *json
);
int descobrirTamanhoMatriz(
    size_t *bufferNumeroLinhas,
    size_t *bufferNumeroColunas,
    cJSON *const matriz
);


cJSON *carregarJSON(const char *jsonFilePath)
{
    FILE *fp;
    errno_t fErro = fopen_s(
        &fp,
        jsonFilePath,
        "rb"
    );
    if (fErro != 0) {
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

// REVIEW - testar
int carregarParametros(
    float *h,
    float *k,
    int   **posicoesGrade,
    int   **conexoes,
    float **condicoesContorno,
    const cJSON *json
)
{
    if (json == NULL) {
        return 1;
    }

    const cJSON *_h = cJSON_GetObjectItemCaseSensitive(json, "x_dist");
    const cJSON *_k = cJSON_GetObjectItemCaseSensitive(json, "y_dist");
    cJSON *const _posicoes = cJSON_GetObjectItemCaseSensitive(json, "ij_pos");
    cJSON *const _conexoes = cJSON_GetObjectItemCaseSensitive(json, "connect");
    cJSON *const _condCont = cJSON_GetObjectItemCaseSensitive(json, "boundary_condition");

    if (
        !cJSON_IsNumber(_h) ||
        !cJSON_IsNumber(_k) ||
        !cJSON_IsArray(_posicoes) ||
        !cJSON_IsArray(_conexoes) ||
        !cJSON_IsArray(_condCont)
    ) {
        return 2;
    }
//
    size_t linhasPosicoes = 0, colunasPosicoes = 0;
    size_t linhasConexoes = 0, colunasConexoes = 0;
    size_t linhasCondCont = 0, colunasCondCont = 0;

    int erro;
    erro = descobrirTamanhoMatriz(&linhasPosicoes, &colunasPosicoes, _posicoes);
    if (erro) {
        return 3;
    }
    erro = descobrirTamanhoMatriz(&linhasConexoes, &colunasConexoes, _conexoes);
    if (erro) {
        return 4;
    }
    erro = descobrirTamanhoMatriz(&linhasCondCont, &colunasCondCont, _condCont);
    if (erro) {
        return 5;
    }
//
    int *bufferPosicoes = (int *)     malloc(linhasPosicoes*colunasPosicoes*sizeof(int));
    if (bufferPosicoes == NULL) {
        return 6;
    }
    int   *bufferConexoes = (int *)   malloc(linhasConexoes*colunasConexoes*sizeof(int));
    if (bufferConexoes == NULL) {
        return 7;
    }
    float *bufferCondCont = (float *) malloc(linhasCondCont*colunasCondCont*sizeof(float));
    if (bufferCondCont == NULL) {
        return 8;
    }
//
    *h = cJSON_GetNumberValue(_h);
    *k = cJSON_GetNumberValue(_k);
    *posicoesGrade     = bufferPosicoes;
    *conexoes          = bufferConexoes;
    *condicoesContorno = bufferCondCont;
    return 0;
}

// REVIEW - testar
int descobrirTamanhoMatriz(
    size_t *bufferNumeroLinhas,
    size_t *bufferNumeroColunas,
    cJSON *const matriz
)
{
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

int main(
    int argc,
    char const *argv[]
)
{
    if (argc != 2) {
        puts(">>> This program take 1 argument.");
        puts(">>> Run again passing the path to an archive .json containing values to:\n");
        puts("    <x_dist>, <y_dist>, <ij_pos>, <connect>, <boundary_condiditon>");
        exit(1);
    }

    cJSON *json = NULL;
    json = carregarJSON(argv[1]);

    float h = 0.0, k = 0.0;
    int   *posicoesGrade, *conexoes;
    float *condicoesContorno;
    carregarParametros(&h, &k, &posicoesGrade, &conexoes, &condicoesContorno, json);
    cJSON_Delete(json);
    return 0;
}
