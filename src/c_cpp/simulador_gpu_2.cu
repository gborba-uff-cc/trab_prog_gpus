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
    size_t *linhasPosicoesGrade,
    size_t *colunassPosicoesGrade,
    int   **conexoes,
    size_t *linhasConexoes,
    size_t *colunassConexoes,
    float **condicoesContorno,
    size_t *linhasContorno,
    size_t *colunasContorno,
    const cJSON *json
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

int carregarParametros(
    float *h,
    float *k,
    int   **posicoesGrade,
    size_t *linhasPosicoesGrade,
    size_t *colunassPosicoesGrade,
    int   **conexoes,
    size_t *linhasConexoes,
    size_t *colunassConexoes,
    float **condicoesContorno,
    size_t *linhasContorno,
    size_t *colunasContorno,
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
    size_t _linhasPosicoes = 0, _colunasPosicoes = 0;
    size_t _linhasConexoes = 0, _colunasConexoes = 0;
    size_t _linhasCondCont = 0, _colunasCondCont = 0;

    int erro;
    erro = descobrirTamanhoMatriz(&_linhasPosicoes, &_colunasPosicoes, _posicoes);
    if (erro) {
        return 3;
    }
    erro = descobrirTamanhoMatriz(&_linhasConexoes, &_colunasConexoes, _conexoes);
    if (erro) {
        return 4;
    }
    erro = descobrirTamanhoMatriz(&_linhasCondCont, &_colunasCondCont, _condCont);
    if (erro) {
        return 5;
    }
//
    int   *bufferPosicoes = (int *)   malloc(_linhasPosicoes*_colunasPosicoes*sizeof(int));
    if (bufferPosicoes == NULL) {
        return 6;
    }
    int   *bufferConexoes = (int *)   malloc(_linhasConexoes*_colunasConexoes*sizeof(int));
    if (bufferConexoes == NULL) {
        return 7;
    }
    float *bufferCondCont = (float *) malloc(_linhasCondCont*_colunasCondCont*sizeof(float));
    if (bufferCondCont == NULL) {
        return 8;
    }
    copiarMatrizIntJsonParaArray(bufferPosicoes, _posicoes, _linhasPosicoes, _colunasPosicoes);
    copiarMatrizIntJsonParaArray(bufferConexoes, _conexoes, _linhasConexoes, _colunasConexoes);
    copiarMatrizFloatJsonParaArray(bufferCondCont, _condCont, _linhasCondCont, _colunasCondCont);
//
    *h = cJSON_GetNumberValue(_h);
    *k = cJSON_GetNumberValue(_k);
    *posicoesGrade         = bufferPosicoes;
    *linhasPosicoesGrade   = _linhasPosicoes;
    *colunassPosicoesGrade = _colunasPosicoes;
    *conexoes          = bufferConexoes;
    *linhasConexoes    = _linhasConexoes;
    *colunassConexoes  = _colunasConexoes;
    *condicoesContorno = bufferCondCont;
    *linhasContorno    = _linhasCondCont;
    *colunasContorno   = _colunasCondCont;

    return 0;
}

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
    int   *posicoesGrade = NULL;
    size_t linhasPosicoesGrade, colunasPosicoesGrade;
    int   *conexoes = NULL;
    size_t linhasConexoes, colunasConexoes;
    float *condicoesContorno = NULL;
    size_t linhasCondCont, colunasCondCont;
    carregarParametros(
        &h, &k,
        &posicoesGrade, &linhasPosicoesGrade, &colunasPosicoesGrade,
        &conexoes, &linhasConexoes, &colunasConexoes,
        &condicoesContorno, &linhasCondCont, &colunasCondCont,
        json
    );

    cJSON_Delete(json);
    free(posicoesGrade);
    free(conexoes);
    free(condicoesContorno);
    return 0;
}
