#include <stdlib.h>
#include <stdio.h>

#include "simulador_gpu_2.h"


int main(
	int argc,
	char const *argv[]
) {
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

	resolverPvcTemperatura(
		&resultado,
		&elementosResultado,
		h, k,
		conexoes, linhasConexoes, colunasConexoes,
		condicoesContorno, linhasCondCont, colunasCondCont
	);

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