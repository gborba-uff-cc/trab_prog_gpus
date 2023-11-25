#ifndef SIMULADOR_GPU_2_H
#define SIMULADOR_GPU_2_H

#include <cuda_runtime.h>

#include "cJSON.h"

// =============================================================================
# define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, 0); }

// =============================================================================

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

int resolverPvcTemperatura(
    float **h_ptrResultado,
    size_t *tamanhoResultado,
    float h,
    float k,
    int *h_posicoesGrade,
    size_t linhasPosicoesGrade,
    size_t colunasPosicoesGrade,
    int *h_conexoes,
    size_t linhasConexoes,
    size_t colunasConexoes,
    float *h_condicoesContorno,
    size_t linhasCondCont,
    size_t colunasCondCont
);

__global__ void k_preencherSistemaEquacoes(
    // matriz bidimensional [n,2]
    const int *bufferConexoes,
    const size_t linhasConexoes,
    const size_t colunasConexoes,
    //
    const float *bufferCondCont,
    const size_t linhasCondCont,
    const size_t colunasCondCont,
    //
    const float *coeficientes_CDEBC,
    const size_t linhasCoeficientes,
    // matriz bidimensional [n,n]
    float *bufferA,
    const size_t linhasA,
    const size_t colunasA,
    // array bidimensional [n]
    float *bufferB,
    const size_t linhasB
);

__global__ void k_imporCondicoesContorno(
    float *bufferA,
    const size_t linhasA,
    const size_t colunasA,
    float *bufferB,
    const size_t linhasB,
    const float *bufferCondCont,
    const size_t linhasCondCont,
    const size_t colunasCondCont
);

int resolverSistemaEquacoes(
    float **h_ptrResultado,
    size_t *linhasX,
    float *d_bufferA,
    const size_t linhasA,
    const size_t colunasA,
    float *d_bufferB,
    const size_t linhasB
);

#endif