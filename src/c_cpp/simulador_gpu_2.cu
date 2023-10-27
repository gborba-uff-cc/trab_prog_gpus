#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
int salvarCsvSimulador2(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t colunasImagem,
    const char* const caminhoNomeArquivo
);
int salvarJsonSimulador2(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t colunasImagem,
    const char* const caminhoNomeArquivo
);
cJSON *gerarResultadoJsonSimulador2(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t colunasImagem
);
void concatenarStrings(
    char **str,
    const char* const bufferStr1,
    const char* const bufferStr2
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
int salvarCsvSimulador2(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t linhasImagem,
    const char* const caminhoNomeArquivo
) {
    if (bufferDominio == NULL) {
        return 1;
    }
    else if (bufferImagem == NULL) {
        return 2;
    }
    else if (linhasDominio !=linhasImagem) {
        return 3;
    }


    FILE *f = NULL;
    errno_t erro = 0;

    erro = fopen_s(&f, caminhoNomeArquivo, "wb");
    if (erro) {
        fclose(f);
        return 4;
    }

    char linha[55] = {0};

    fputs("iPos,jPos,Temperature\n", f);
    for (size_t i=0; i<linhasDominio; i++) {
        for (size_t j=0; j<sizeof(linha);j++) {
            linha[j] = '\0';
        }
        erro = sprintf_s(
            linha,
            sizeof(linha),
            "%d,%d,%.15f\n",
            bufferDominio[i*colunasDominio],
            bufferDominio[i*colunasDominio+1],
            bufferImagem[i]
        );
        if (erro < 1) {
            return 5;
        }
        fputs(linha, f);
    }

    fclose(f);
    return 0;
}
int salvarJsonSimulador2(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t linhasImagem,
    const char* const caminhoNomeArquivo
) {
    if (bufferDominio == NULL) {
        return 1;
    }
    else if (bufferImagem == NULL) {
        return 2;
    }
    else if (linhasDominio !=linhasImagem) {
        return 3;
    }

    cJSON *json = gerarResultadoJsonSimulador2(
        bufferDominio,
        linhasDominio,
        colunasDominio,
        bufferImagem,
        linhasImagem
    );

    if (json == NULL) {
        return 4;
    }

    FILE *f = NULL;
    errno_t erro = 0;

    erro = fopen_s(&f, caminhoNomeArquivo, "wb");

    if (erro) {
        fclose(f);
        return 5;
    }

    const char *conteudoJson = cJSON_PrintUnformatted(json);
    fputs(conteudoJson, f);

    fclose(f);
    return 0;
}
cJSON *gerarResultadoJsonSimulador2(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t linhasImagem
) {
    cJSON *resultado = cJSON_CreateObject();
    cJSON *_dominio  = cJSON_AddArrayToObject(resultado, "domain");
    for (size_t i=0; _dominio != NULL && i<linhasDominio; i++) {
        cJSON *anEntry = cJSON_CreateIntArray(
            &bufferDominio[i*colunasDominio],
            colunasDominio
        );
        if (anEntry == NULL) {
            cJSON_Delete(_dominio);
            break;
        }
        else {
            cJSON_AddItemToArray(_dominio, anEntry);
        }
    }
    cJSON *_imagem = cJSON_CreateFloatArray(bufferImagem, linhasImagem);
    cJSON_AddItemToObject(resultado, "image", _imagem);

    if (_dominio == NULL || _imagem == NULL) {
        cJSON_Delete(resultado);
        resultado = NULL;
    }

    return resultado;
}
void concatenarStrings(char **str, const char* const bufferStr1, const char* const bufferStr2) {
    size_t tamanhoBuffer = strlen(bufferStr1) + strlen(bufferStr2) + 1;
    char* const buffer = (char *) malloc(tamanhoBuffer);

    strcpy_s(buffer, tamanhoBuffer, bufferStr1);
    strcat_s(buffer, tamanhoBuffer, bufferStr2);

    *str = buffer;
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
    float *resultado = NULL;
    size_t elementosResultado = 0;

    char caminhoNomeArquivo[] = ".\\testeSimulador2";
    char *caminhoNomeArquivoJson;
    char *caminhoNomeArquivoCsv;
    concatenarStrings(
        &caminhoNomeArquivoJson,
        caminhoNomeArquivo,
        ".json"
    );
    concatenarStrings(
        &caminhoNomeArquivoCsv,
        caminhoNomeArquivo,
        ".csv"
    );
    salvarJsonSimulador2(
        (int*) posicoesGrade,
        linhasPosicoesGrade,
        colunasPosicoesGrade,
        resultado,
        elementosResultado,
        caminhoNomeArquivoJson
    );
    salvarCsvSimulador2(
        (int*) posicoesGrade,
        linhasPosicoesGrade,
        colunasPosicoesGrade,
        resultado,
        elementosResultado,
        caminhoNomeArquivoCsv
    );

    cJSON_Delete(json);
    free(posicoesGrade);
    free(conexoes);
    free(condicoesContorno);
    free(caminhoNomeArquivoJson);
    free(caminhoNomeArquivoCsv);
    return 0;
}
