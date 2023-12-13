#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "simulador_gpu_1.h"

#include <device_launch_parameters.h>

// =============================================================================
int executaSimulador1(
	const char * const caminhoArquivoEntrada
) {
	size_t numeroElementos = 0;
	size_t numeroPassos = 0;
	float tamanhoPasso =0.0;
	float *x0= NULL, *y0 = NULL;
	float dSpringX = 0.0, dSpringY = 0.0;
	float massa = 0.0;
	float constanteElastica = 0.0;
	float *forcasExternasX = NULL, *forcasExternasY = NULL;
	int *restricoesX = NULL, *restricoesY = NULL;
	int *conexoes = NULL;
	size_t colunasConexoes = 0;

	float *resultadoX = NULL;
	float *resultadoY = NULL;
	size_t elementosResultado = 0;

	cJSON *json = NULL;
	json = carregarJSON(caminhoArquivoEntrada);

	carregarParametrosSimulador1(
		&numeroElementos,
		&numeroPassos, &tamanhoPasso,
		&x0, &y0,
		&dSpringX, &dSpringY,
		&massa, &constanteElastica,
		&forcasExternasX, &forcasExternasY,
		&restricoesX, &restricoesY,
		&conexoes, &colunasConexoes,
		json
	);

	cJSON_Delete(json);

	resolverPvi2LeiNewton(
		&resultadoX,
		&resultadoY,
		&elementosResultado,
		95,
		numeroElementos,
		x0, y0,
		dSpringX, dSpringY,
		tamanhoPasso, numeroPassos,
		massa, constanteElastica,
		forcasExternasX, forcasExternasY,
		restricoesX, restricoesY,
		conexoes, colunasConexoes
	);

	free(x0);
	free(y0);
	free(forcasExternasX);
	free(forcasExternasY);
	free(restricoesX);
	free(restricoesY);
	free(conexoes);

	char *stringHorario = NULL;
	char *caminhoNomeArquivo = NULL;
	char *caminhoNomeArquivoJson = NULL;

	dateTimeAsString(&stringHorario);
	concatenarStrings(&caminhoNomeArquivo, ".\\sim2LeiNewton_", stringHorario);
	concatenarStrings(
		&caminhoNomeArquivoJson,
		caminhoNomeArquivo,
		".json"
	);

	salvarJsonSimulador1(resultadoX, elementosResultado, caminhoNomeArquivoJson);

	free(resultadoX);
	free(resultadoY);

	free(stringHorario);
	free(caminhoNomeArquivo);
	free(caminhoNomeArquivoJson);

	return 0;
}

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
	*restricoesX = buffersRestricoes[0];
	*restricoesY = buffersRestricoes[1];
	*conexoes = bufferConexoes;
	*colunasConexoes = _colunasConexoes;

// libera buffers não usados se eles existirem
	for (size_t i=2; i<_colunasCoordenadas; i++) {
		free(buffersCoordenadas[i]);
	}
	free(buffersCoordenadas);

	for (size_t i=2; i<_colunasForcas; i++) {
		free(buffersForcas[i]);
	}
	free(buffersForcas);

	for (size_t i=2; i<_colunasRestricoes; i++) {
		free(buffersRestricoes[i]);
	}
	free(buffersRestricoes);

	return 0;
}

int salvarJsonSimulador1(
    const float *bufferImagem,
    const size_t colunasImagem,
    const char* const caminhoNomeArquivo
) {
	if (bufferImagem == NULL) {
		return 1;
	}

	cJSON *json = gerarResultadoJsonSimulador1(
		bufferImagem,
		colunasImagem
	);

	if (json == NULL) {
		return 2;
	}

	FILE *f = NULL;
	f = fopen(caminhoNomeArquivo, "wb");

	if (!f) {
		return 3;
	}

	const char *conteudoJson = cJSON_PrintUnformatted(json);
	fputs(conteudoJson, f);

	fclose(f);
	return 0;
}

cJSON *gerarResultadoJsonSimulador1(
    const float *bufferImagem,
    const size_t colunasImagem
) {
	cJSON *resultado = cJSON_CreateObject();

	cJSON *_imagem = cJSON_CreateFloatArray(bufferImagem, colunasImagem);
	cJSON_AddItemToObject(resultado, "resultado", _imagem);

	if (_imagem == NULL) {
		cJSON_Delete(resultado);
		resultado = NULL;
	}

	return resultado;
}

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
) {
	float *deslocamentoXParticulaObservada = NULL;
	float *deslocamentoYParticulaObservada = NULL;

	float *d_x0 = NULL;
	float *d_y0 = NULL;
	float *d_forcasExternasX = NULL;
	float *d_forcasExternasY = NULL;
	int *d_restricoesX = NULL;
	int *d_restricoesY = NULL;
	int *d_conexoes = NULL;

	float *d_aceleracaoX = NULL;
	float *d_aceleracaoY = NULL;
	float *d_velocidadeX = NULL;
	float *d_velocidadeY = NULL;
	float *d_deslocamentoX = NULL;
	float *d_deslocamentoY = NULL;
	float *d_forcasInternasX = NULL;
	float *d_forcasInternasY = NULL;

	float *d_deslocamentoXParticulaObservada = NULL;
	float *d_deslocamentoYParticulaObservada = NULL;

	size_t bytesX0 = sizeof(float)*numeroElementos;
	size_t bytesY0 = bytesX0;
	size_t bytesForcasExternasX = bytesX0;
	size_t bytesForcasExternasY = bytesX0;
	size_t bytesRestricoesX = sizeof(int)*numeroElementos;
	size_t bytesRestricoesY = bytesRestricoesX;
	size_t bytesConexoes = sizeof(int)*numeroElementos*colunasConexoes;
	size_t bytesAceleracaoX = bytesX0;
	size_t bytesAceleracaoY = bytesX0;
	size_t bytesVelocidadeX = bytesX0;
	size_t bytesVelocidadeY = bytesX0;
	size_t bytesDeslocamentoX = bytesX0;
	size_t bytesDeslocamentoY = bytesX0;
	size_t bytesForcasInternasX = bytesX0;
	size_t bytesForcasInternasY = bytesX0;
	size_t bytesDeslocamentoXParticulaObservada = sizeof(float)*numeroPassos;
	size_t bytesDeslocamentoYParticulaObservada = bytesDeslocamentoXParticulaObservada;

	cudaMalloc(
		(void **) &d_x0,
		bytesX0
	);
	cudaMalloc(
		(void **) &d_y0,
		bytesY0
	);
	cudaMalloc(
		(void **) &d_forcasExternasX,
		bytesForcasExternasX
	);
	cudaMalloc(
		(void **) &d_forcasExternasY,
		bytesForcasExternasY
	);
	cudaMalloc(
		(void **) &d_restricoesX,
		bytesRestricoesX
	);
	cudaMalloc(
		(void **) &d_restricoesY,
		bytesRestricoesY
	);
	cudaMalloc(
		(void **) &d_conexoes,
		bytesConexoes
	);

	cudaMalloc(
		(void **) &d_aceleracaoX,
		bytesAceleracaoX
	);
	cudaMalloc(
		(void **) &d_aceleracaoY,
		bytesAceleracaoY
	);
	cudaMalloc(
		(void **) &d_velocidadeX,
		bytesVelocidadeX
	);
	cudaMalloc(
		(void **) &d_velocidadeY,
		bytesVelocidadeY
	);
	cudaMalloc(
		(void **) &d_deslocamentoX,
		bytesDeslocamentoX
	);
	cudaMalloc(
		(void **) &d_deslocamentoY,
		bytesDeslocamentoY
	);
	cudaMalloc(
		(void **) &d_forcasInternasX,
		bytesForcasInternasX
	);
	cudaMalloc(
		(void **) &d_forcasInternasY,
		bytesForcasInternasY
	);
	cudaMalloc(
		(void **) &d_deslocamentoXParticulaObservada,
		bytesDeslocamentoXParticulaObservada
	);
	cudaMalloc(
		(void **) &d_deslocamentoYParticulaObservada,
		bytesDeslocamentoYParticulaObservada
	);

	cudaMemcpyAsync(d_x0, x0, bytesX0, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_y0, y0, bytesY0, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_forcasExternasX, forcasExternasX, bytesForcasExternasX, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_forcasExternasY, forcasExternasY, bytesForcasExternasY, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_restricoesX, restricoesX    , bytesRestricoesX, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_restricoesY, restricoesY    , bytesRestricoesY, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_conexoes, conexoes , bytesConexoes, cudaMemcpyHostToDevice);
	cudaMemsetAsync(d_aceleracaoX, 0, bytesAceleracaoX);
	cudaMemsetAsync(d_aceleracaoY, 0, bytesAceleracaoY);
	cudaMemsetAsync(d_velocidadeX, 0, bytesVelocidadeX);
	cudaMemsetAsync(d_velocidadeY, 0, bytesVelocidadeY);
	cudaMemsetAsync(d_deslocamentoX, 0, bytesDeslocamentoX);
	cudaMemsetAsync(d_deslocamentoY, 0, bytesDeslocamentoY);
	cudaMemsetAsync(d_forcasInternasX, 0, bytesForcasInternasX);
	cudaMemsetAsync(d_forcasInternasY, 0, bytesForcasInternasY);
	cudaMemsetAsync(d_deslocamentoXParticulaObservada, 0, bytesDeslocamentoXParticulaObservada);
	cudaMemsetAsync(d_deslocamentoYParticulaObservada, 0, bytesDeslocamentoYParticulaObservada);

	const size_t tamanhoBloco = 32*30;
	const size_t numeroBlocos = (numeroElementos-1)/tamanhoBloco+1;
	cudaDeviceSynchronize();

	for (size_t passoSimulacao=0; passoSimulacao<numeroPassos; passoSimulacao++) {
		k_leapfrogSegundaLeiNewton1Pt<<<numeroBlocos, tamanhoBloco>>>(
			d_deslocamentoXParticulaObservada, d_deslocamentoYParticulaObservada, numeroElementos,
			passoSimulacao, particulaObservada,
			d_x0, d_y0,
			dSpringX, dSpringY,
			tamanhoPasso, numeroPassos, massa, constanteElastica,
			d_forcasExternasX, d_forcasExternasY,
			d_restricoesX, d_restricoesY,
			d_conexoes, colunasConexoes,
			d_aceleracaoX, d_aceleracaoY,
			d_velocidadeX, d_velocidadeY,
			d_deslocamentoX, d_deslocamentoY,
			d_forcasInternasX, d_forcasInternasY
		);
		gpuErrchk(cudaPeekAtLastError());
		cudaDeviceSynchronize();

		k_algoritmoContato<<<numeroBlocos, tamanhoBloco>>>(
			numeroElementos,
			d_x0,
			d_y0,
			dSpringX,
			dSpringY,
			d_restricoesX,
			d_restricoesY,
			d_conexoes,
			colunasConexoes,
			d_deslocamentoX,
			d_deslocamentoY,
			d_forcasInternasX,
			d_forcasInternasY,
			constanteElastica
		);
		gpuErrchk(cudaPeekAtLastError());
		cudaDeviceSynchronize();

		k_leapfrogSegundaLeiNewton2Pt<<<numeroBlocos, tamanhoBloco>>>(
			d_deslocamentoXParticulaObservada,
			d_deslocamentoYParticulaObservada,
			numeroElementos,
			passoSimulacao,
			particulaObservada,
			tamanhoPasso,
			massa,
			d_forcasExternasX,
			d_forcasExternasY,
			d_aceleracaoX,
			d_aceleracaoY,
			d_velocidadeX,
			d_velocidadeY,
			d_deslocamentoX,
			d_deslocamentoY,
			d_forcasInternasX,
			d_forcasInternasY
		);
		gpuErrchk(cudaPeekAtLastError());
		cudaDeviceSynchronize();
	}
	gpuErrchk(cudaDeviceSynchronize());

	deslocamentoXParticulaObservada = (float *) malloc(bytesDeslocamentoXParticulaObservada);
	deslocamentoYParticulaObservada = (float *) malloc(bytesDeslocamentoYParticulaObservada);

	cudaMemcpyAsync(
		deslocamentoXParticulaObservada,
		d_deslocamentoXParticulaObservada,
		bytesDeslocamentoXParticulaObservada,
		cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(
		deslocamentoYParticulaObservada,
		d_deslocamentoYParticulaObservada,
		bytesDeslocamentoYParticulaObservada,
		cudaMemcpyDeviceToHost
	);
	cudaDeviceSynchronize();

	*ptrResultadoX = deslocamentoXParticulaObservada;
	*ptrResultadoY = deslocamentoYParticulaObservada;
	*elementosResultado = numeroPassos;
	return 0;
}

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
) {
	size_t gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= numeroElementos) {
		return;
	}

	if (numeroPassoSimulacao == 0) {
		aceleracaoX[gIdx] = (forcasExternasX[gIdx] - forcasInternasX[gIdx])/massa;
		aceleracaoY[gIdx] = (forcasExternasY[gIdx] - forcasInternasY[gIdx])/massa;
	}

	velocidadeX[gIdx] += aceleracaoX[gIdx]*(tamanhoPasso*0.5);
	velocidadeY[gIdx] += aceleracaoY[gIdx]*(tamanhoPasso*0.5);
	deslocamentoX[gIdx] += velocidadeX[gIdx]*tamanhoPasso;
	deslocamentoY[gIdx] += velocidadeY[gIdx]*tamanhoPasso;

	return;
}

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
) {
	size_t gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= numeroElementos) {
		return;
	}

	forcasInternasX[gIdx] = 0;
	forcasInternasY[gIdx] = 0;

	deslocamentoX[gIdx] *= 1-restricoesX[gIdx];
	deslocamentoY[gIdx] *= 1-restricoesY[gIdx];

	float xPasso = x0[gIdx] + deslocamentoX[gIdx];
	float yPasso = y0[gIdx] + deslocamentoY[gIdx];

	int vizinho = 0;
	float xPassoVizinho = 0.0, yPassoVizinho = 0.0, normaPosicao = 0.0,
	deltaX = 0.0, deltaY = 0.0,
	deformacaoMolaX = 0.0, deformacaoMolaY = 0.0;
	// primera coluna da matriz conexoes no arquivo tem o numero de
	// vizinhos do elemento; essa implementação considera 4 vizinhos
	for (size_t i=1; i<colunasConexoes; i++) {
		vizinho = conexoes[gIdx*colunasConexoes+i];
		// a matriz de conexoes no arquivo tem a numeracao iniciando em 1
		if (vizinho > 0) {
			vizinho -= 1;
			xPassoVizinho = x0[vizinho]+deslocamentoX[vizinho];
			yPassoVizinho = y0[vizinho]+deslocamentoY[vizinho];

			deltaX = xPasso-xPassoVizinho;
			deltaY = yPasso-yPassoVizinho;

			normaPosicao = hypotf(deltaX, deltaY);

			// deformacao da mola que une os elementos
			deformacaoMolaX = normaPosicao-2*dSpringX;
			deformacaoMolaY = normaPosicao-2*dSpringY;

			deltaX = deformacaoMolaX*deltaX/normaPosicao;
			deltaY = deformacaoMolaY*deltaY/normaPosicao;

			forcasInternasX[gIdx] += constanteElastica*deltaX;
			forcasInternasY[gIdx] += constanteElastica*deltaY;
		}
	}

	return;
}

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
) {
	size_t gIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (gIdx >= numeroElementos) {
		return;
	}

	//armazena resultado
	if (gIdx == particulaObservada) {
		deslocamentoXParticulaObservada[numeroPassoSimulacao] = deslocamentoX[gIdx];
		deslocamentoYParticulaObservada[numeroPassoSimulacao] = deslocamentoY[gIdx];
	}

	aceleracaoX[gIdx] = (forcasExternasX[gIdx] - forcasInternasX[gIdx])/massa;
	aceleracaoY[gIdx] = (forcasExternasY[gIdx] - forcasInternasY[gIdx])/massa;
	// velocidade no fim do passo
	velocidadeX[gIdx] += aceleracaoX[gIdx]*tamanhoPasso*0.5;
	velocidadeY[gIdx] += aceleracaoY[gIdx]*tamanhoPasso*0.5;

	return;
}
