# Trabalho final da disciplina

semestre: 2023-2

status: objetivo finalizado

## Objetivo geral

Escrever código em CUDA para resolver algum problema com o uso de GPUs.

## Objetivo escolhido

Modificar o simulador desenvolvido durante as aulas de programação científica para que ele tire proveito da GPU durante para a realização dos cálculos em lugar da CPU.

## Requisitos originais

- Linguagem Julia
  - Pacotes:
    - Dates
    - JSON
    - SparseArrays
- Linguagem Python 3
  - Pacotes:
    - matplotlib
    - numpy
    - PyOpenGL
    - PyQt5
    - jsonschema

## Observações

- O código inicial é o que foi escrito por mim durante as aulas da disciplina de 'Programação Científica' com a finalidade de servir como avaliação.

- O simulador de PVI gera a saída apenas para o eixo x do elemento de numero 95, se quiser o resultado para outro elemento o código precisa ser alterado.

- O modelador de PVI não tem interface para aplicar forças externas nem restrções, alterações deve ser realizadas diretamente no json com a nuvem de pontos gerado.