<#
.SINOPSIS
    Muestra el avance de la cola de experimentos en el servidor del laboratorio.

.DESCRIPCION
    Se conecta por SSH al alias `phantom` y reporta en qué experimento va la cola,
    las cifras que ha producido hasta ahora, el estado de la tarjeta grafica y del
    disco, y si algo ha fallado. Cuando la cola termina, ejecuta ademas el
    verificador de indicadores.

    No modifica nada en el servidor: solo lee registros.

.PARAMETER Seguir
    Repite la consulta cada `Intervalo` segundos hasta que se interrumpa con
    Ctrl+C. Sin este parametro consulta una sola vez.

.PARAMETER Intervalo
    Segundos entre consultas cuando se usa -Seguir. Por defecto 300 (cinco
    minutos). No conviene bajar mucho: cada corrida dura unos quince minutos, asi
    que consultar mas a menudo no muestra nada nuevo.

.PARAMETER Detalle
    Muestra todas las lineas de metricas del experimento en curso en vez de las
    ultimas ocho.

.EJEMPLO
    .\scripts\estado_cola.ps1
    Consulta una vez y termina.

.EJEMPLO
    .\scripts\estado_cola.ps1 -Seguir
    Consulta cada cinco minutos hasta que se interrumpa.

.EJEMPLO
    .\scripts\estado_cola.ps1 -Seguir -Intervalo 900 -Detalle
    Consulta cada quince minutos mostrando todas las metricas.
#>
[CmdletBinding()]
param(
    [switch] $Seguir,
    [int]    $Intervalo = 300,
    [switch] $Detalle
)

$ErrorActionPreference = 'Continue'

# El guion remoto viaja codificado en base64. Se hace asi por dos razones que se
# comprobaron fallando: al canalizarlo a `bash -s`, PowerShell 5.1 antepone una
# marca de orden de bytes que el shell remoto interpreta como parte del primer
# mandato ("echo: command not found"), y ademas convierte los saltos de linea a
# CRLF, con lo que cada linea llega con un retorno de carro pegado ($'whoami\r').
# En base64 no hay marca, ni saltos de linea, ni comillas que escapar en dos
# niveles, de modo que el guion llega byte a byte tal como se escribio.
function Get-GuionRemoto {
    param([int] $Lineas)

    $plantilla = @'
cd /data/calegre/Tesis || exit 1

echo "FECHA::$(date '+%Y-%m-%d %H:%M:%S')"

# Que experimento va, segun el registro de la cola.
echo "SECCION::COLA"
if [ -f data/reports/experimentos/cola.log ]; then
    grep -E "^\[20" data/reports/experimentos/cola.log | tail -6
else
    echo "  (todavia no hay registro de cola)"
fi

# El experimento en curso: se deduce del proceso vivo, no del registro, porque
# entre dos experimentos encolados el registro ya anuncio el siguiente pero aun
# no ha empezado a producir nada.
ACTUAL=$(ps -u "$(id -un)" -o cmd= | awk '!/awk/ && /experimentos\/e[0-9]/ {print $3; exit}')
echo "SECCION::ACTUAL"
if [ -n "$ACTUAL" ]; then
    echo "  $ACTUAL"
    REG="data/reports/experimentos/$(basename "$ACTUAL" .py).log"
    if [ -f "$REG" ]; then
        echo "SECCION::METRICAS"
        grep -E "F1=|eps=|\[OK|\[FALLA|veredicto|orden:|piso |semilla" "$REG" | tail -LINEAS_
    fi
else
    echo "  (ningun experimento en ejecucion)"
fi

echo "SECCION::PROCESOS"
ps -u "$(id -un)" -o pid=,etime=,cmd= | awk '!/awk/ && (/cola_servidor/ || /experimentos\/e[0-9]/) {printf "  pid %-8s activo %-12s %s\n", $1, $2, $4}'

echo "SECCION::MAQUINA"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | awk '{printf "  GPU %s\n", $0}'
df -h /data | tail -1 | awk '{printf "  disco: %s libres de %s (%s usado)\n", $4, $2, $5}'

echo "SECCION::FALLOS"
if [ -s data/reports/experimentos/FALLOS ]; then
    cat data/reports/experimentos/FALLOS
elif grep -q "FALL" data/reports/experimentos/cola.log 2>/dev/null; then
    grep "FALL" data/reports/experimentos/cola.log | tail -3
else
    echo "  ninguno"
fi

echo "SECCION::TERMINADA"
if [ -f data/reports/experimentos/COLA_COMPLETA ]; then
    echo "  SI, desde $(cat data/reports/experimentos/COLA_COMPLETA)"
    echo "SECCION::COBERTURA"
    PYTHONPATH=src .venv/bin/python scripts/experimentos/consolidar_iov.py 2>/dev/null | grep -E "^\s+\[|resultados cubiertos|  - R"
else
    echo "  no"
fi
'@
    return $plantilla.Replace('LINEAS_', $Lineas.ToString())
}

function Show-Bloque {
    param([string] $Titulo, [string[]] $Lineas, [string] $Color = 'Gray')

    if ($Lineas.Count -eq 0) { return }
    Write-Host ""
    Write-Host $Titulo -ForegroundColor Cyan
    foreach ($l in $Lineas) {
        if ($l.Trim().Length -gt 0) { Write-Host $l -ForegroundColor $Color }
    }
}

function Show-Estado {
    param([int] $Lineas)

    $guion = Get-GuionRemoto -Lineas $Lineas
    $guion = $guion -replace "`r`n", "`n"
    $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($guion))
    $salida = ssh -o BatchMode=yes -o ConnectTimeout=20 phantom "echo $b64 | base64 -d | bash" 2>&1

    if ($LASTEXITCODE -ne 0 -and -not $salida) {
        Write-Host "No se pudo conectar al servidor. Comprobar 'ssh phantom'." -ForegroundColor Red
        return
    }

    # Se reparte la salida por marcadores en vez de por posicion: si el guion
    # remoto crece, el reparto sigue siendo correcto.
    $bloques = @{}
    $actual = 'CABECERA'
    $fecha = ''
    foreach ($linea in $salida) {
        $texto = [string]$linea
        if ($texto -like 'FECHA::*') { $fecha = $texto.Substring(7); continue }
        if ($texto -like 'SECCION::*') { $actual = $texto.Substring(9); $bloques[$actual] = @(); continue }
        if (-not $bloques.ContainsKey($actual)) { $bloques[$actual] = @() }
        $bloques[$actual] += $texto
    }

    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkGray
    Write-Host " Cola de experimentos - servidor phantom - $fecha" -ForegroundColor White
    Write-Host ("=" * 72) -ForegroundColor DarkGray

    Show-Bloque "Registro de la cola" $bloques['COLA']
    Show-Bloque "Experimento en curso" $bloques['ACTUAL'] 'Yellow'
    Show-Bloque "Metricas producidas" $bloques['METRICAS'] 'Green'
    Show-Bloque "Procesos activos" $bloques['PROCESOS']
    Show-Bloque "Maquina" $bloques['MAQUINA']

    $fallos = $bloques['FALLOS']
    if ($fallos -and ($fallos -join '') -notmatch 'ninguno') {
        Show-Bloque "FALLOS" $fallos 'Red'
    } else {
        Show-Bloque "Fallos" $fallos 'DarkGray'
    }

    $terminada = ($bloques['TERMINADA'] -join '')
    if ($terminada -match 'SI') {
        Show-Bloque "Cola terminada" $bloques['TERMINADA'] 'Green'
        Show-Bloque "Cobertura de indicadores" $bloques['COBERTURA'] 'Green'
        return $true
    }
    return $false
}

$lineas = 8
if ($Detalle) { $lineas = 40 }

if (-not $Seguir) {
    Show-Estado -Lineas $lineas | Out-Null
    exit 0
}

Write-Host "Consultando cada $Intervalo segundos. Ctrl+C para terminar." -ForegroundColor DarkGray
while ($true) {
    $completa = Show-Estado -Lineas $lineas
    if ($completa -eq $true) {
        Write-Host ""
        Write-Host "La cola termino. Fin del seguimiento." -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds $Intervalo
}
