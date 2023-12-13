#ifndef SIMULADOR_GPU_1_H
#define SIMULADOR_GPU_1_H

/*
Calculo de PVI para segunda lei de Newton
*/

#include <cuda_runtime.h>

#include "cJSON.h"

// =============================================================================
int executaSimulador1(
	const char * const caminhoArquivoEntrada
);

int carregarParametrosSimulador1(
	size_t *numeroElementos,
	size_t *numeroPassos,
	float *tamanhoPasso,
	float **x0,
	float **y0,
	float *dSpringX,
	float *dSpringY,
	float *massa,
	float *constanteElastica,
	float **forcasExternasX,
	float **forcasExternasY,
	int **restricoesX,
	int **restricoesY,
	int **conexoes,
	size_t *colunasConexoes,
	const cJSON *json
);

int salvarJsonSimulador1(
    const float *bufferImagem,
    const size_t colunasImagem,
    const char* const caminhoNomeArquivo
);

cJSON *gerarResultadoJsonSimulador1(
    const float *bufferImagem,
    const size_t colunasImagem
);

int resolverPvi2LeiNewton(
	float **ptrResultadoX,
	float **ptrResultadoY,
	size_t *elementosResultado,
	size_t particulaObservada,
	size_t numeroElementos,
	float *x0,
	float *y0,
	float dSpringX,
	float dSpringY,
	float tamanhoPasso,
	size_t numeroPassos,
	float massa,
	float constanteElastica,
	float *forcasExternasX,
	float *forcasExternasY,
	int *restricoesX,
	int *restricoesY,
	int *conexoes,
	size_t colunasConexoes
);

__global__ void k_leapfrogSegundaLeiNewton1Pt(
	float *deslocamentoXParticulaObservada,
	float *deslocamentoYParticulaObservada,
	size_t numeroElementos,
	size_t numeroPassoSimulacao,
	size_t particulaObservada,
	float *x0,
	float *y0,
	float dSpringX,
	float dSpringY,
	float tamanhoPasso,
	size_t numeroPassos,
	float massa,
	float constanteElastica,
	float *forcasExternasX,
	float *forcasExternasY,
	int *restricoesX,
	int *restricoesY,
	int *conexoes,
	size_t colunasConexoes,
	float *aceleracaoX,
	float *aceleracaoY,
	float *velocidadeX,
	float *velocidadeY,
	float *deslocamentoX,
	float *deslocamentoY,
	float *forcasInternasX,
	float *forcasInternasY
);

__global__ void k_algoritmoContato(
	size_t numeroElementos,
	float *x0,
	float *y0,
	float dSpringX,
	float dSpringY,
	int *restricoesX,
	int *restricoesY,
	int *conexoes,
	size_t colunasConexoes,
	float *deslocamentoX,
	float *deslocamentoY,
	float *forcasInternasX,
	float *forcasInternasY,
	float constanteElastica
);

__global__ void k_leapfrogSegundaLeiNewton2Pt(
	float *deslocamentoXParticulaObservada,
	float *deslocamentoYParticulaObservada,
	size_t numeroElementos,
	size_t numeroPassoSimulacao,
	size_t particulaObservada,
	float tamanhoPasso,
	float massa,
	float *forcasExternasX,
	float *forcasExternasY,
	float *aceleracaoX,
	float *aceleracaoY,
	float *velocidadeX,
	float *velocidadeY,
	float *deslocamentoX,
	float *deslocamentoY,
	float *forcasInternasX,
	float *forcasInternasY
);
#endif