#include <stdlib.h>
#include <stdio.h>

#include "utils.h"
#include "simulador_gpu_1.h"
#include "simulador_gpu_2.h"


int main(
	int argc,
	char const *argv[]
) {
	if (argc != 2) {
		puts(\
"Para executar o algoritmo do simulador que resolve o PVI da segunda lei de Newton\n"
"execute novamente passando o caminho para um arquivo .json como:\n"
"    simulador --pvi caminho/arquivo.json\n"
"\n"
"o arquivo json caminho/arquivo.json contem valores para: <step_size>, \n"
"<number_steps>, <particle_mass>, <particle_hardness>, <particle_half_xSize>\n"
"<particle_half_ySize>, <particle_coords>, <particle_external_force>, \n"
"<particle_restricted>, <particle_connection>\n"
"\n"
"onde:\n"
"    <step_size>, <number_steps>, <particle_mass>, <particle_hardness>, \n"
"      <particle_half_xSize> <particle_half_ySize> devem ser um valor numerico\n"
"    <particle_coords> matriz onde cada linha tem a posicao x e y de um \n"
"      elemento\n"
"    <particle_external_force> matriz onde cada linha tem os componentes x e y\n"
"      do vetor forca que atua sobre o elemento que a linha representa\n"
"    <particle_restricted> matriz onde cada linha indica se a componente x e y\n"
"      da particula esta restringida de ser movida (1 sim, 0 nao)\n"
"    <particle_connection> matriz onde cada linha na primeira coluna c indica\n"
"      quantos vizinhos o elemento representado pelo indice da linha tem \n"
"      conectado a ele seguido de 4 valores onde apenas os c primeiros numeros\n"
"      serão considerados\n"
	);
		puts(\
"Para executar o algoritmo do simulador que resolve o PVC da conducao de calor\n"
"execute novamente passando o caminho para um arquivo .json como:\n"
"    simulador --pvc caminho/arquivo.json\n"
"\n"
"o arquivo json caminho/arquivo.json contem valores para\n"
"    <x_dist>, <y_dist>, <ij_pos>, <connect>, <boundary_condiditon>\n"
"\n"
"onde:\n"
"    <x_dist> e <y_dist> devem ser um valor numerico\n"
"    <ij_pos> matriz onde cada linha tem o indice da posicao x e y de um \n"
"      elemento (indices iniciam em 1)\n"
"    <connect> matriz onde cada linha contem indices para os vizinhos do elemento\n"
"    <boundary_condiditon> matriz onde cada linha tem uma indicacao de que o elemento\n"
"      tem uma condicao de contorno imposta e qual o valor imposto\n"
	);
		exit(1);
	}

	const char *idSimuladorSelecionado = argv[1];
	const char *caminhoArquivoEntrada = argv[2];
	char *caminhoArquivoSaida = NULL;
	char *stringHorario = NULL;
	char *caminhoNomeArquivo = NULL;

	if (argc > 3) {
		// copia o caminho passado
		concatenarStrings(&caminhoArquivoSaida, argv[3], "");
	}
	dateTimeAsString(&stringHorario);

	int falha = 0;
	if (strcasecmp(idSimuladorSelecionado, "pvi") == 0) {
		if (!caminhoArquivoSaida) {
			concatenarStrings(&caminhoArquivoSaida, ".\\sim1_2LeiNewton_", stringHorario);
		}
		falha = executaSimulador1(caminhoArquivoEntrada, caminhoArquivoSaida);
	}
	else if (strcasecmp(idSimuladorSelecionado, "pvc") == 0) {
		if (!caminhoArquivoSaida) {
			concatenarStrings(&caminhoNomeArquivo, ".\\sim2_conducaoTermica_", stringHorario);
		}
		falha = executaSimulador2(caminhoArquivoEntrada, caminhoArquivoSaida);
	}
	else {
		fprintf(stderr, "O simulador requisitado não foi encontrado");
		falha = 1;
	}

	free(caminhoArquivoSaida);
	free(stringHorario);
	free(caminhoNomeArquivo);

	return falha;
}