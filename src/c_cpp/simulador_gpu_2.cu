#include <stdio.h>
#include <stdlib.h>

#include "cJSON.h"

#define TIPO_CARACTERE_ARQUIVO char
#define MAXIMO_BYTES_ARQUIVO_JSON 1000001
#define MAXIMO_CARACTERES_BUFFER_ARQUIVO MAXIMO_BYTES_ARQUIVO_JSON / sizeof(TIPO_CARACTERE_ARQUIVO)


cJSON *carregarJSON(const char *jsonFilePath);


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
    int *posicoesGrade, *conexoes, *condicoesContorno;
    carregarParametros(&h, &k, &posicoesGrade, &conexoes, &condicoesContorno, json);
    cJSON_Delete(json);
    return 0;
}
