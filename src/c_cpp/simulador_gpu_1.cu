#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "simulador_gpu_1.h"

#include <device_launch_parameters.h>

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
) {
	if (json == NULL) {
		return 1;
	}

	cJSON *const _coordenadasParticulas = cJSON_GetObjectItemCaseSensitive(json, "particle_coords");
	const cJSON *_particleHalfXSize = cJSON_GetObjectItemCaseSensitive(json, "particle_half_xSize");
	const cJSON *_particleHalfYSize = cJSON_GetObjectItemCaseSensitive(json, "particle_half_ySize");
	const cJSON *_tamanhoPasso = cJSON_GetObjectItemCaseSensitive(json, "step_size");
	const cJSON *_numeroPassos = cJSON_GetObjectItemCaseSensitive(json, "number_steps");
	const cJSON *_massa = cJSON_GetObjectItemCaseSensitive(json, "particle_mass");
	const cJSON *_constanteElastica = cJSON_GetObjectItemCaseSensitive(json, "particle_hardness");
	cJSON *const _forcasExternas = cJSON_GetObjectItemCaseSensitive(json, "particle_external_force");
	cJSON *const _restricoes = cJSON_GetObjectItemCaseSensitive(json, "particle_restricted");
	cJSON *const _conexoes = cJSON_GetObjectItemCaseSensitive(json, "particle_connection");

	if (
		!cJSON_IsArray(_coordenadasParticulas) ||
		!cJSON_IsNumber(_particleHalfXSize) ||
		!cJSON_IsNumber(_particleHalfYSize) ||
		!cJSON_IsNumber(_tamanhoPasso) ||
		!cJSON_IsNumber(_numeroPassos) ||
		!cJSON_IsNumber(_massa) ||
		!cJSON_IsNumber(_constanteElastica) ||
		!cJSON_IsArray(_forcasExternas) ||
		!cJSON_IsArray(_restricoes) ||
		!cJSON_IsArray(_conexoes)
	) {
		return 2;
	}
//
	size_t _linhasCoordenadas = 0, _colunasCoordenadas = 0;
	size_t _linhasForcas = 0, _colunasForcas = 0;
	size_t _linhasRestricoes = 0, _colunasRestricoes = 0;
	size_t _linhasConexoes = 0, _colunasConexoes = 0;

	int erro;
	erro = descobrirTamanhoMatriz(&_linhasCoordenadas, &_colunasCoordenadas, _coordenadasParticulas);
	if (erro) {
		return 3;
	}
	erro = descobrirTamanhoMatriz(&_linhasForcas, &_colunasForcas, _forcasExternas);
	if (erro) {
		return 4;
	}
	erro = descobrirTamanhoMatriz(&_linhasRestricoes, &_colunasRestricoes, _restricoes);
	if (erro) {
		return 5;
	}
	erro = descobrirTamanhoMatriz(&_linhasConexoes, &_colunasConexoes, _conexoes);
	if (erro) {
		return 6;
	}

	if (!(
		_linhasCoordenadas == _linhasForcas && \
		_linhasCoordenadas == _linhasRestricoes && \
		_linhasCoordenadas == _linhasConexoes)) {
		return 7;
	}
//
	int erroAlocacao = 0;
	// array segurando outros arrays
	float **buffersCoordenadas = (float **) malloc(_colunasCoordenadas*sizeof(float *));
	if (!buffersCoordenadas) {
		erroAlocacao = 1;
	}
	for (size_t i=0; !erroAlocacao && i<_colunasCoordenadas; i++) {
		buffersCoordenadas[i] = NULL;
	}
	for (size_t i=0; !erroAlocacao && i<_colunasCoordenadas; i++) {
		buffersCoordenadas[i] = (float *) malloc(_linhasCoordenadas*sizeof(float));
		if (!buffersCoordenadas[i]) {
			fprintf(stderr, "apenas %ld de %ld array(s) de coordenadas foram alocados", i, _colunasCoordenadas);
			erroAlocacao = 1;
		}
	}
	float **buffersForcas = (float **) malloc(_colunasForcas*sizeof(float *));
	for (size_t i=0; i<_colunasForcas; i++) {
		buffersForcas[i] = NULL;
	}
	for (size_t i=0; !erroAlocacao && i<_colunasForcas; i++) {
		buffersForcas[i] = (float *) malloc(_linhasForcas*sizeof(float));
		if (!buffersForcas[i]) {
			fprintf(stderr, "apenas %ld de %ld array(s) de forcas externas foram alocados", i, _colunasForcas);
			erroAlocacao = 1;
		}
	}
	int **buffersRestricoes = (int **) malloc(_colunasRestricoes*sizeof(int *));
	for (size_t i=0; i<_colunasRestricoes; i++) {
		buffersRestricoes[i] = NULL;
	}
	for (size_t i=0; !erroAlocacao && i<_colunasRestricoes; i++) {
		buffersRestricoes[i] = (int *) malloc(_linhasRestricoes*sizeof(float));
		if (!buffersRestricoes[i]) {
			fprintf(stderr, "apenas %ld de %ld array(s) de restricoes foram alocados", i, _colunasRestricoes);
			erroAlocacao = 1;
		}
	}
/*
// conexoes como array de arrays
	int **buffersConexoes = (int **) malloc(_colunasConexoes*sizeof(int *));
	for (size_t i=0; i<_colunasConexoes; i++) {
		buffersConexoes[i] = NULL;
	}
	for (size_t i=0; !erroAlocacao && i<_colunasConexoes; i++) {
		buffersConexoes[i] = (int *) malloc(_linhasConexoes*sizeof(int));
		if (!buffersConexoes[i]) {
			fprintf(stderr, "apenas %ld de %ld array(s) de restricoes foram alocados", i, _colunasConexoes);
			erroAlocacao = 1;
		}
	}
*/
	int *bufferConexoes = (int *) malloc(_linhasConexoes*_colunasConexoes*sizeof(int *));
	if (!bufferConexoes) {
		erroAlocacao = 1;
	}

	if (erroAlocacao) {
		for (size_t i=0; i<_colunasCoordenadas; i++) {
			free(buffersCoordenadas[i]);
		}
		free(buffersCoordenadas);

		for (size_t i=0; i<_colunasForcas; i++) {
			free(buffersForcas[i]);
		}
		free(buffersForcas);

		for (size_t i=0; i<_colunasRestricoes; i++) {
			free(buffersRestricoes[i]);
		}
		free(buffersRestricoes);

		free(bufferConexoes);
/*
// conexoes como array de arrays
		for (size_t i=0; i<_colunasConexoes; i++) {
			free(buffersConexoes[i]);
		}
		free(buffersConexoes);
*/

		return 8;
	}
	// FIXME precisa copiar cada coluna da matriz em um array
	copiarColunasMatrizJsonParaArrays_float(
		buffersCoordenadas,
		_coordenadasParticulas, _linhasCoordenadas, _colunasCoordenadas
	);
	copiarColunasMatrizJsonParaArrays_float(
		buffersForcas,
		_forcasExternas, _linhasForcas, _colunasForcas
	);
	copiarColunasMatrizJsonParaArrays_int(
		buffersRestricoes,
		_restricoes, _linhasRestricoes, _colunasRestricoes
	);
/*
// conexoes como array de arrays
	copiarMatrizIntJsonParaArray(
		buffersConexoes,
		_conexoes, _linhasConexoes, _colunasConexoes
	);
*/
	copiarMatrizIntJsonParaArray(
		bufferConexoes,
		_conexoes, _linhasConexoes, _colunasConexoes
	);
//
	*numeroElementos = _linhasCoordenadas;
	*x0 = buffersCoordenadas[0];
	*y0 = buffersCoordenadas[1];
	*dSpringX = (float) cJSON_GetNumberValue(_particleHalfXSize);
	*dSpringY = (float) cJSON_GetNumberValue(_particleHalfYSize);
	*tamanhoPasso = (float) cJSON_GetNumberValue(_tamanhoPasso);
	*numeroPassos = (size_t) cJSON_GetNumberValue(_numeroPassos);
	*massa = (float) cJSON_GetNumberValue(_massa);
	*constanteElastica = (float) cJSON_GetNumberValue(_constanteElastica);
	*forcasExternasX = buffersForcas[0];
	*forcasExternasY = buffersForcas[1];
	*restrictedX = buffersRestricoes[0];
	*restrictedY = buffersRestricoes[1];
	*conexoes = bufferConexoes;
	*colunasConexoes = _colunasConexoes;

	return 0;
}
