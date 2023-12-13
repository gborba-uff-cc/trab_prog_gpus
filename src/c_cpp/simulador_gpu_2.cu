#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "simulador_gpu_2.h"

#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// =============================================================================
int executaSimulador2(
	const char * const caminhoArquivoEntrada
) {
	cJSON *json = NULL;
	json = carregarJSON(caminhoArquivoEntrada);

	float h = 0.0, k = 0.0;
	int   *posicoesGrade = NULL;
	size_t linhasPosicoesGrade, colunasPosicoesGrade;
	int   *conexoes = NULL;
	size_t linhasConexoes, colunasConexoes;
	float *condicoesContorno = NULL;
	size_t linhasCondCont, colunasCondCont;
	carregarParametrosSimulador2(
		&h, &k,
		&posicoesGrade, &linhasPosicoesGrade, &colunasPosicoesGrade,
		&conexoes, &linhasConexoes, &colunasConexoes,
		&condicoesContorno, &linhasCondCont, &colunasCondCont,
		json
	);
	float *resultado = NULL;
	size_t elementosResultado = 0;

	resolverPvcTemperatura(
		&resultado,
		&elementosResultado,
		h, k,
		conexoes, linhasConexoes, colunasConexoes,
		condicoesContorno, linhasCondCont, colunasCondCont
	);

	char *stringHorario = NULL;
	char *caminhoNomeArquivo = NULL;
	char *caminhoNomeArquivoJson = NULL;
	char *caminhoNomeArquivoCsv = NULL;

	dateTimeAsString(&stringHorario);
	concatenarStrings(&caminhoNomeArquivo, ".\\sim1ConducaoTermica_", stringHorario);
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
	free(resultado);

	free(stringHorario);
	free(caminhoNomeArquivo);
	free(caminhoNomeArquivoJson);
	free(caminhoNomeArquivoCsv);
	return 0;
}

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
	f = fopen(caminhoNomeArquivo, "wb");
	if (!f) {
		return 4;
	}

	char linha[55] = {0};

	int erro;
	fputs("iPos,jPos,Temperature\n", f);
	for (size_t i=0; i<linhasDominio; i++) {
		for (size_t j=0; j<sizeof(linha);j++) {
			linha[j] = '\0';
		}
		erro = sprintf(
			linha,
			"%d,%d,%.15f\n",
			bufferDominio[i*colunasDominio],
			bufferDominio[i*colunasDominio+1],
			bufferImagem[i]
		);
		if (erro < 0) {
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
	f = fopen(caminhoNomeArquivo, "wb");

	if (!f) {
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

// resolve o problema do valor de contorno de temperatura bidimensional.
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
) {
	const float hkRatio = h/k;
	const float kCentro = 2*(hkRatio*hkRatio+1);
	const float kDireita = -1.0;
	const float kEsquerda = kDireita;
	const float kAbaixo = -(hkRatio*hkRatio);
	const float kAcima = kAbaixo;
	const float coeficientes_CentroDEBC[] = {
		kCentro, kDireita, kEsquerda, kAbaixo, kAcima
	};

	int *d_conexoes = NULL;
	size_t bytesConexoes = sizeof(int)*linhasConexoes*colunasConexoes;
	float *d_condicoesContorno = NULL;
	size_t bytesCondCont = sizeof(float)*linhasCondCont*colunasCondCont;
	float *d_coeficientes_CentroDEBC = NULL;
	const size_t linhasCoeficientes = 5;
	const size_t bytesCoeficientes = sizeof(float)*linhasCoeficientes;
	float *d_A = NULL;
	const size_t linhasA = linhasConexoes;
	const size_t colunasA = linhasConexoes;
	const size_t bytesA = sizeof(float)*linhasA*colunasA;
	float *d_b = NULL;
	size_t linhasB = linhasConexoes;
	size_t bytesB = sizeof(float)*linhasB;

	cudaMalloc(
		(void **) &d_conexoes,
		bytesConexoes
	);
	cudaMalloc(
		(void **) &d_condicoesContorno,
		bytesCondCont
	);
	cudaMalloc(
		(void **) &d_coeficientes_CentroDEBC,
		bytesCoeficientes
	);
	cudaMalloc(
		(void **) &d_A,
		bytesA
	);
	cudaMalloc(
		(void **) &d_b,
		bytesB
	);

	cudaMemcpyAsync(
		d_conexoes,
		h_conexoes,
		bytesConexoes,
		cudaMemcpyHostToDevice
	);
	cudaMemcpyAsync(
		d_condicoesContorno,
		h_condicoesContorno,
		bytesCondCont,
		cudaMemcpyHostToDevice
	);
	cudaMemcpyAsync(
		d_coeficientes_CentroDEBC,
		coeficientes_CentroDEBC,
		bytesCoeficientes,
		cudaMemcpyHostToDevice
	);

	// SECTION - prepara sistema de equacoes
	size_t tamanhoProblema = linhasConexoes;
	size_t tamanhoBloco = 32;
	int numeroBlocos = (tamanhoProblema-1)/tamanhoBloco+1;

	cudaDeviceSynchronize();
	k_preencherSistemaEquacoes<<<numeroBlocos, tamanhoBloco>>>(
		d_conexoes, linhasConexoes, colunasConexoes,
		d_condicoesContorno, linhasCondCont, colunasCondCont,
		d_coeficientes_CentroDEBC, linhasCoeficientes,
		d_A, linhasA, colunasA,
		d_b, linhasB
	);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	tamanhoProblema = linhasA*colunasA;
	tamanhoBloco = 128;
	numeroBlocos = (tamanhoProblema-1)/tamanhoBloco+1;
	k_imporCondicoesContorno<<<numeroBlocos, tamanhoBloco>>>(
		d_A, linhasA, colunasA,
		d_b, linhasB,
		d_condicoesContorno, linhasCondCont, colunasCondCont
	);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// !SECTION

	resolverSistemaEquacoes(
		h_ptrResultado, tamanhoResultado,
		&d_A, linhasA, colunasA,
		d_b, linhasB
	);

	cudaFree(d_conexoes);
	cudaFree(d_condicoesContorno);
	cudaFree(d_coeficientes_CentroDEBC);
	cudaFree(d_A);
	cudaFree(d_b);
	return 0;
}

__global__ void k_preencherSistemaEquacoes(
	const int *bufferConexoes,
	const size_t linhasConexoes,
	const size_t colunasConexoes,
	const float *bufferCondCont,
	const size_t linhasCondCont,
	const size_t colunasCondCont,
	const float *coeficientes_CDEBC,
	const size_t linhasCoeficientes,
	float *bufferA,
	const size_t linhasA,
	const size_t colunasA,
	float *bufferB,
	const size_t linhasB
) {
	const size_t idxB = threadIdx.x;
	const size_t idxG = blockIdx.x * blockDim.x + idxB;
	const size_t idxLinha = idxG;

	if (idxG >= linhasA) {
		return;
	}

	const size_t iBaseBufferA = idxLinha*colunasA;
	const size_t iBaseConexoes = idxLinha*colunasConexoes;
	size_t iBaseCondCont = 0;
	size_t idxVizinho = 0;
	float existeCondCont = 0;
	bufferA[iBaseBufferA + idxLinha] = coeficientes_CDEBC[0];
	for (size_t numVizinho=1; numVizinho<linhasCoeficientes; numVizinho++) {
		idxVizinho = bufferConexoes[iBaseConexoes + (numVizinho-1)];
		if (idxVizinho > 0) {
			idxVizinho -= 1;
			iBaseCondCont = idxVizinho*colunasCondCont;
			existeCondCont = bufferCondCont[iBaseCondCont + 0];
			if (existeCondCont) {
				bufferB[idxLinha] += bufferCondCont[iBaseCondCont + 1];
			}
			else {
				bufferA[iBaseBufferA + idxVizinho] = coeficientes_CDEBC[numVizinho];
			}
		}
	}

	return;
}
__global__ void k_imporCondicoesContorno(
	float *bufferA,
	const size_t linhasA,
	const size_t colunasA,
	float *bufferB,
	const size_t linhasB,
	const float *bufferCondCont,
	const size_t linhasCondCont,
	const size_t colunasCondCont
) {
	const size_t idxB = threadIdx.x;
	const size_t idxG = blockIdx.x * blockDim.x + idxB;

	if (idxG >= linhasA * colunasA) {
		return;
	}

	size_t i = idxG / colunasA;
	size_t j = idxG % colunasA;
	size_t iBaseCondCont = i*colunasCondCont;

	if (!bufferCondCont[iBaseCondCont + 0]) {
		return;
	}
	if (i == j) {
		bufferA[idxG] = 1.0;
	}
	else {
		bufferA[idxG] = 0.0;
	}
	if (j == 0) {
		bufferB[i] = bufferCondCont[iBaseCondCont + 1];
	}
}
int resolverSistemaEquacoes(
	float **h_ptrResultado,
	size_t *linhasX,
	float **d_bufferA,
	const size_t linhasA,
	const size_t colunasA,
	float *d_bufferB,
	const size_t linhasB
) {
// SECTION - transpoe a matriz (de row-major para column-major)
	size_t bytesAT = sizeof(float)*linhasA*colunasA;
	// tranposta da matriz A
	float *d_AT = NULL;
	cudaMalloc(
		(void **) &d_AT,
		bytesAT
	);
	cudaMemset(d_AT, 0, bytesAT);

	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	float alpha = 1.0;
	float beta = 0.0;
	// NOTE - muda buffer de row-major para column-major;
	cublasStatus_t status = cublasSgeam(
		cublasHandle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		linhasA, linhasA,
		&alpha, *d_bufferA, linhasA,
		&beta, d_AT, linhasA,
		d_AT, linhasA
	);
	cudaDeviceSynchronize();
	if (status != CUBLAS_STATUS_SUCCESS) {
		return 1;
	}

	cublasDestroy(cublasHandle);
	cudaFree(*d_bufferA);
	*d_bufferA = d_AT;
// !SECTION

// SECTION - tratando como matriz densa
	// NOTE - fatoracao LU
	cusolverDnHandle_t cusolverHandler;
	cusolverDnCreate(&cusolverHandler);

	cusolverDnParams_t parametros;
	cusolverDnCreateParams(&parametros);

	cusolverDnSetAdvOptions(parametros, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);

	size_t d_tamanhoWorkspace = 0;
	size_t h_tamanhoWorkspace = 0;
	void *d_workspace = NULL;
	void *h_workspace = NULL;

	cusolverDnXgetrf_bufferSize(
		cusolverHandler, parametros, linhasA, colunasA,
		CUDA_R_32F, d_AT, colunasA,
		CUDA_R_32F, &d_tamanhoWorkspace, &h_tamanhoWorkspace
	);

	cudaMalloc((void **) &d_workspace, d_tamanhoWorkspace);
	h_workspace = malloc(h_tamanhoWorkspace);
	if (!h_workspace) {
		cudaFree(d_workspace);
		cusolverDnDestroyParams(parametros);
		cusolverDnDestroy(cusolverHandler);
		return 2;
	}

	size_t tamanhoIpiv = sizeof(int64_t)*linhasA;
	int64_t *d_ipiv;
	size_t tamanhoInformacao = sizeof(int);
	int* d_informacao;
	int h_informacao;
	cudaMalloc((void **) &d_informacao, tamanhoInformacao);
	cudaMalloc((void **) &d_ipiv, tamanhoIpiv);
	cusolverDnXgetrf(
		cusolverHandler, parametros, linhasA, colunasA,
		CUDA_R_32F, d_AT, colunasA, d_ipiv, CUDA_R_32F,
		d_workspace, d_tamanhoWorkspace,
		h_workspace, h_tamanhoWorkspace,
		d_informacao
	);
	cudaMemcpy(
		&h_informacao, d_informacao, tamanhoInformacao, cudaMemcpyDeviceToHost
	);
	if (h_informacao < 0 || h_informacao > 0) {
		int ret = 0;
		if (h_informacao < 0) {
			fprintf(stderr, "parametro na posicao %d está errado", -h_informacao);
			ret = 3;
		}
		else {
			fprintf(stderr, "LU factorization failed, U[,%d,%d]==0.0\n", h_informacao, h_informacao);
			ret = 4;
		}
		cudaFree(d_workspace);
		cudaFree(d_informacao);
		cudaFree(d_ipiv);
		cusolverDnDestroyParams(parametros);
		cusolverDnDestroy(cusolverHandler);
		return ret;
	}

	// NOTE - soluciona sistema
	cusolverDnXgetrs(
		cusolverHandler, parametros, CUBLAS_OP_N, linhasA, 1, /* nrhs */
		CUDA_R_32F, d_AT, colunasA,
		d_ipiv,
		CUDA_R_32F, d_bufferB, linhasB,
		d_informacao
	);
	cudaMemcpy(
		&h_informacao, d_informacao, tamanhoInformacao, cudaMemcpyDeviceToHost
	);
	if (h_informacao < 0) {
		printf("parametro na posicao %d está errado", -h_informacao);
		cudaFree(d_workspace);
		cudaFree(d_informacao);
		cudaFree(d_ipiv);
		cusolverDnDestroyParams(parametros);
		cusolverDnDestroy(cusolverHandler);
		return 5;
	}

	float *aux = (float *) malloc(sizeof(float)*linhasB);
	cudaMemcpy(aux, d_bufferB, sizeof(float)*linhasB, cudaMemcpyDeviceToHost);
	*h_ptrResultado = aux;
	*linhasX = linhasB;

	free(h_workspace);
	cudaFree(d_workspace);
	cudaFree(d_informacao);
	cudaFree(d_ipiv);
	cusolverDnDestroyParams(parametros);
	cusolverDnDestroy(cusolverHandler);
// !SECTION

	return 0;
}
