#ifndef SIMULADOR_GPU_2_H
#define SIMULADOR_GPU_2_H

/*
Calculo de PVC para problema de distribuicao de calor em uma "chapa"
*/

#include <cuda_runtime.h>

#include "cJSON.h"

// =============================================================================
int executaSimulador2(
	const char * const caminhoArquivoEntrada,
	const char * const caminhoArquivoSaida
);

int carregarParametrosSimulador2(
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

int resolverPvcTemperatura(
    float **h_ptrResultado,
    size_t *tamanhoResultado,
    float h,
    float k,
    int *h_conexoes,
    size_t linhasConexoes,
    size_t colunasConexoes,
    float *h_condicoesContorno,
    size_t linhasCondCont,
    size_t colunasCondCont
);

__global__ void k_preencherSistemaEquacoes(
    // matriz [n][4] row-major
    const int *bufferConexoes,
    const size_t linhasConexoes,
    const size_t colunasConexoes,
    // matriz [n][2] row-major
    const float *bufferCondCont,
    const size_t linhasCondCont,
    const size_t colunasCondCont,
    // array de coeficientes para cinco posicoes
    const float *coeficientes_CDEBC,
    const size_t linhasCoeficientes,
    // array [n][n]
    float *bufferA,
    const size_t linhasA,
    const size_t colunasA,
    // array [n]
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
    float **d_bufferA,
    const size_t linhasA,
    const size_t colunasA,
    float *d_bufferB,
    const size_t linhasB
);

#endif