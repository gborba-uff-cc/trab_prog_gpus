#ifndef SIMULADOR_GPU_1_H
#define SIMULADOR_GPU_1_H

/*
Calculo de PVI para segunda lei de Newton
*/

#include <cuda_runtime.h>

#include "cJSON.h"

// =============================================================================
int carregarParametrosSimulador1(
	size_t *numeroElementos,
	float **x0,
	float **y0,
	float *dSpringX,
	float *dSpringY,
	float *tamanhoPasso,
	size_t *numeroPassos,
	float *massa,
	float *constanteElastica,
	float **forcasExternasX,
	float **forcasExternasY,
	int **restrictedX,
	int **restrictedY,
	int **conexoes,
	size_t *colunasConexoes,
	const cJSON *json
);

int salvarCsvSimulador1(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t colunasImagem,
    const char* const caminhoNomeArquivo
);

int salvarJsonSimulador1(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t colunasImagem,
    const char* const caminhoNomeArquivo
);

cJSON *gerarResultadoJsonSimulador1(
    // matriz bidimensional
    const int *bufferDominio,
    const size_t linhasDominio,
    const size_t colunasDominio,
    // array
    const float *bufferImagem,
    const size_t colunasImagem
);

int resolverPvi(
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

#endif