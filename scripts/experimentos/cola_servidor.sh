#!/usr/bin/env bash
# Cola de los ocho experimentos (E0 a E7) en el servidor del laboratorio.
#
# Estrictamente SECUENCIAL y en una sola tarjeta. No es una preferencia: si dos
# corridas compartieran GPU, los tiempos dejarían de ser comparables y la memoria
# disponible variaría entre condiciones del mismo experimento, que es una
# diferencia de método leyéndose como una de arquitectura.
#
# El orden importa, pero por dependencia de ARTEFACTOS y no de conclusiones: E3
# reevalua sobre textos degradados los puntos de control que entrenan E1 y E2, y
# E6 mide la latencia de los que entrena E4. Ningun experimento hereda de otro una
# eleccion de ganadora --eso se retiro a proposito--: cada uno compara todas las
# arquitecturas que su pregunta requiere. Ejecutarlos en desorden deja informes
# incompletos sin emitir error.
#
#   bash scripts/experimentos/cola_servidor.sh
#
# Escribe una marca al terminar: los guiones que consumen los resultados deben
# esperar a que exista, y no a que no haya procesos de Python. Se comprobó que lo
# segundo falla, porque entre dos corridas encoladas hay un instante sin proceso
# y una espera basada en eso arranca antes de tiempo.

set -u  # sin -e: un experimento que falle no debe cancelar los demás

# La cola se ejecuta desde una COPIA de si misma, y no desde el fichero del
# repositorio. La razon es concreta y costo una corrida de veintidos horas: bash
# lee un guion por desplazamiento de bytes MIENTRAS lo ejecuta, de modo que
# editarlo a media corrida desplaza lo que aun no ha leido. Se comprobo: al
# anadir un experimento a la lista, bash retomo en una posicion desfasada, cayo
# en mitad de `${PIPESTATUS[0]}` y el bucle murio con un error de sintaxis
# despues de E6. Los seis experimentos anteriores habian terminado bien, pero no
# se ejecutaron ni la consolidacion ni las figuras, y la marca de fin no se
# escribio. Con la copia, editar el original durante la corrida es inocuo.
if [ "${COLA_DESDE_COPIA:-0}" != "1" ]; then
    copia="$(mktemp -t cola_servidor.XXXXXX.sh)"
    cat "$0" > "$copia"
    chmod +x "$copia"
    # La copia vive en /tmp, asi que `dirname $0` ya no sirve para localizar el
    # proyecto: se le pasa la raiz resuelta antes de saltar.
    COLA_RAIZ="$(cd "$(dirname "$0")/../.." && pwd)" \
        COLA_DESDE_COPIA=1 bash "$copia" "$@"
    rc=$?
    rm -f "$copia"
    exit "$rc"
fi

cd "${COLA_RAIZ:-$(dirname "$0")/../..}" || exit 1
export PYTHONPATH="src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Sin esto, Python almacena la salida y el registro no muestra nada hasta que
# el guion termina: en la prueba local pareció colgado catorce minutos cuando
# en realidad ya había entrenado la primera variante.
export PYTHONUNBUFFERED=1

# Interprete EXPLICITO. Con `python` a secas la cola resolvia al Python del
# sistema al lanzarse desde una sesion no interactiva --que es como corre
# desacoplada-- y los siete experimentos fallaban en un segundo con "Unable to
# find a usable engine" al leer el parquet, porque ese interprete no tiene
# pyarrow. Se puede sobreescribir con PY=... para otro entorno.
PY_BIN="${PY:-.venv/bin/python}"
if [ ! -x "$PY_BIN" ]; then
    echo "No existe el interprete $PY_BIN. Indicar otro con PY=/ruta" >&2
    exit 1
fi

REG="data/reports/experimentos"
mkdir -p "$REG"
MARCA="$REG/COLA_COMPLETA"
rm -f "$MARCA"

EPOCAS="${EPOCAS:-3}"

# Cada corrida escribe dos puntos de control: el mejor --solo pesos, unos 260
# MiB-- y el ultimo, que ademas lleva el estado del optimizador y ronda los 777
# MiB. Con las variantes y semillas de los siete experimentos son decenas de
# gigabytes: conviene comprobar el disco antes.
libre=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc "0-9")
if [ -n "$libre" ] && [ "$libre" -lt 80 ]; then
    echo "AVISO: quedan ${libre} GiB libres; la cola puede necesitar mas de 80."
fi

fallos=0

# E7 va despues de E5 porque su lectura se apoya en la composicion por idioma
# que E5 mide, y antes de E6, que mide latencias en CPU y no debe compartirla.
EXPERIMENTOS="e0_corpus_y_pipeline e1_multimodalidad e2_mecanismo_fusion
              e3_ausencia_ramas e4_generalizacion e5_validez
              e7_codificador_multilingue e6_modelo_optimizado"

# Todos los guiones tienen que compilar ANTES de gastar la primera hora de GPU.
# Un error de sintaxis en el septimo experimento no se manifiesta hasta que le
# toca el turno: se comprobo, y costo dieciocho horas de cola para descubrir que
# E7 no arrancaba. Compilar los ocho tarda un segundo.
echo "Comprobando que los ocho guiones compilan..."
for exp in $EXPERIMENTOS; do
    if ! "$PY_BIN" -m py_compile "scripts/experimentos/${exp}.py"; then
        echo "ABORTA: scripts/experimentos/${exp}.py no compila." >&2
        exit 1
    fi
done
for aux in scripts/experimentos/consolidar_iov.py scripts/figuras/generar_figuras_mv.py; do
    if ! "$PY_BIN" -m py_compile "$aux"; then
        echo "ABORTA: $aux no compila." >&2
        exit 1
    fi
done
echo "  los ocho experimentos y los dos guiones de cierre compilan."

for exp in $EXPERIMENTOS; do
    echo "=============================================================="
    echo "[$(date '+%F %T')] $exp"
    echo "=============================================================="
    if [ "$exp" = "e5_validez" ] || [ "$exp" = "e6_modelo_optimizado" ] \
       || [ "$exp" = "e0_corpus_y_pipeline" ]; then
        "$PY_BIN" -u "scripts/experimentos/${exp}.py" 2>&1 | tee "$REG/${exp}.log"
    else
        "$PY_BIN" -u "scripts/experimentos/${exp}.py" --epocas "$EPOCAS" 2>&1 \
            | tee "$REG/${exp}.log"
    fi
    # El código de salida de una tubería es el del último mandato, que aquí es
    # `tee` y siempre vale cero. Se consulta el del guion.
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%F %T')] $exp FALLÓ (código $rc)"
        fallos=$((fallos + 1))
    else
        echo "[$(date '+%F %T')] $exp terminado"
    fi
done

echo "=============================================================="
if [ "$fallos" -eq 0 ]; then
    date '+%F %T' > "$MARCA"
    echo "COLA COMPLETA: los ocho experimentos terminaron"
    # El verificador de indicadores va DENTRO de la cola: comprobar la cobertura a
    # mano es como se colaron artefactos que describian un corpus anterior.
    # Va PRIMERO porque tambien traslada los historiales de entrenamiento a la
    # carpeta de R1.4, que es de donde salen las curvas de aprendizaje.
    "$PY_BIN" -u scripts/experimentos/consolidar_iov.py 2>&1 | tee "$REG/cobertura_iov.log"
    # Las figuras que exigen los medios de verificacion y que ningun experimento
    # emite: matrices de confusion, curvas ROC y curvas de aprendizaje.
    "$PY_BIN" -u scripts/figuras/generar_figuras_mv.py 2>&1 | tee "$REG/figuras_mv.log"
else
    echo "COLA TERMINADA CON $fallos FALLO(S) · no se escribe la marca"
fi
exit "$fallos"
