#! /bin/pwsh

function tempoExecucao {
    Param (
        [string]$Comando
    )
    $numeroExecucoes = 30
    [System.Object[]]$valoresObtidos = @()
    for ($i=0; $i -lt $numeroExecucoes; $i++) {
        $resultado = Measure-Command { Invoke-Expression $comando }
        $valoresObtidos += $resultado.Ticks
    }
    $media = 0
    for ($i=0; $i -lt $numeroExecucoes; $i++) {
        $media += $valoresObtidos[$i]
    }
    $media = $media/$numeroExecucoes
    $stdDev = 0
    for ($i=0; $i -lt $numeroExecucoes; $i++) {
        $stdDev += [Math]::Pow($valoresObtidos[$i]-$media, 2)
    }
    $stdDev = $stdDev / ($numeroExecucoes-1)
    $stdDev = [Math]::Sqrt($stdDev)

    [PSCustomObject]@{
        'media'        = $media *[Math]::Pow(10,-7)  # ticks = 100 nanosegundos
        'desvioPadrao' = $stdDev*[Math]::Pow(10,-7)  # ticks = 100 nanosegundos
    }
}

function exibeResultado {
    Param(
        [PSCustomObject]$Resultado
    )
    Write-Host $resultado.media 's' ' ' $resultado.desvioPadrao 's' -Separator ''
}



Write-Output 'PVI julia'
Write-Output 'media desvioPadraoAmostral'
$resultado = tempoExecucao -Comando 'julia --project=. ./aula_012_P2_GabrielDaCunhaBorba/q2_leapfrog.jl ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_1_10x10.json'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando 'julia --project=. ./aula_012_P2_GabrielDaCunhaBorba/q2_leapfrog.jl ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_1_50x50.json'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando 'julia --project=. ./aula_012_P2_GabrielDaCunhaBorba/q2_leapfrog.jl ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_3_10x10.json'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando 'julia --project=. ./aula_012_P2_GabrielDaCunhaBorba/q2_leapfrog.jl ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_3_50x50.json'
exibeResultado -Resultado $resultado

Write-Output 'PVI CUDA'
Write-Output 'media desvioPadraoAmostral'
$resultado = tempoExecucao -Comando './output/simulador.bin pvi ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_1_10x10.json'# ./output/q2out_1_10x10'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando './output/simulador.bin pvi ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_1_50x50.json'# ./output/q2out_1_50x50'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando './output/simulador.bin pvi ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_3_10x10.json'# ./output/q2out_1_10x10'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando './output/simulador.bin pvi ./aula_012_P2_GabrielDaCunhaBorba/output/q2in_3_50x50.json'# ./output/q2out_3_50x50'
exibeResultado -Resultado $resultado

Write-Output 'PVC julia'
Write-Output 'media desvioPadraoAmostral'
$resultado = tempoExecucao -Comando 'julia --project=. ./aula_015_P4_GabrielDaCunhaBorba/q1_simulador.jl ./aula_015_P4_GabrielDaCunhaBorba/output/2022y039d22h27m02sIn.json'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando 'julia --project=. ./aula_015_P4_GabrielDaCunhaBorba/q1_simulador.jl ./aula_015_P4_GabrielDaCunhaBorba/output/2022y040d19h03m39sIn.json'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando 'julia --project=. ./aula_015_P4_GabrielDaCunhaBorba/q1_simulador.jl ./aula_015_P4_GabrielDaCunhaBorba/output/2022y040d19h22m41sIn.json'
exibeResultado -Resultado $resultado

Write-Output 'PVC CUDA'
Write-Output 'media desvioPadraoAmostral'
$resultado = tempoExecucao -Comando './output/simulador.bin pvc ./aula_015_P4_GabrielDaCunhaBorba/output/2022y039d22h27m02sIn.json'# ./output/2022y039d22h27m02sOut'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando './output/simulador.bin pvc ./aula_015_P4_GabrielDaCunhaBorba/output/2022y040d19h03m39sIn.json'# ./output/2022y040d19h03m39sIn'
exibeResultado -Resultado $resultado

$resultado = tempoExecucao -Comando './output/simulador.bin pvc ./aula_015_P4_GabrielDaCunhaBorba/output/2022y040d19h22m41sIn.json'# ./output/2022y040d19h22m41sIn'
exibeResultado -Resultado $resultado
