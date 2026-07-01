#!/usr/bin/env python3
"""
Captura de latencia - Tier 2 (Grove Vision AI V2)
IPD441 - Evaluacion YOLOv8n bajo iluminacion variable en edge devices

Uso:
    pip install pyserial --break-system-packages   # si hace falta
    python3 tier2_latency_capture.py --discover     # PASO 1: ver el formato real del log
    python3 tier2_latency_capture.py --port /dev/tty.usbmodemXXXX --gamma 1.0 --n 100

Flujo:
  1. Conectar Grove Vision AI V2 a la Mac via USB-C (sin XIAO).
  2. Cargar el modelo YOLOv8n en el dispositivo desde SenseCraft AI Studio
     (https://sensecraft.seeed.cc/ai/home/) -> confirma que "Vista previa/Iniciar"
     funciona (invoke exitoso), luego dale "Detener" y CIERRA esa pestana/tab
     por completo para liberar el puerto serial (el puerto solo lo puede tener
     abierto un proceso a la vez).
  3. Correr este script. El propio script manda el comando AT+INVOKE=-1,0,1
     (protocolo SSCMA-Micro) para poner al dispositivo en modo de inferencia
     continua - no necesitas el navegador abierto para esto, el modelo ya
     quedo grabado en el dispositivo desde el paso 2.

Referencia del protocolo AT (SSCMA-Micro):
  https://github.com/Seeed-Studio/SSCMA-Micro/blob/main/docs/protocol/at-protocol-en_US.md
  Comando: AT+INVOKE=<N_TIMES,DIFFERED,RESULT_ONLY>\\r
    N_TIMES=-1 (loop infinito), DIFFERED=0, RESULT_ONLY=1
  Evento de respuesta: {"type":1,"name":"INVOKE","data":{"perf":[preprocess,inference,postprocess],"boxes":[...]}}
  perf[1] es el tiempo de INFERENCIA real (lo que este script mide).
"""

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    sys.exit("Falta pyserial. Instalar con: pip install pyserial --break-system-packages")

BAUDRATE_DEFAULT = 921600

# Patrones de respaldo por si el log no viene en JSON limpio.
# Ajustar estos regex despues de ver el output real con --discover.
FALLBACK_PATTERNS = [
    re.compile(r'"perf"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'),  # [pre, infer, post]
    re.compile(r'perf.*?(\d+\.?\d*)\s*ms', re.IGNORECASE),
    re.compile(r'invoke time[:=]\s*(\d+\.?\d*)', re.IGNORECASE),
]


def start_continuous_invoke(ser: "serial.Serial"):
    """Envia el comando AT+INVOKE=-1,0,1 para poner al dispositivo en modo
    de inferencia continua (loop infinito, solo resultados sin imagen).
    Sin esto, el dispositivo se queda callado esperando ordenes."""
    cmd = b"AT+INVOKE=-1,0,1\r"
    ser.write(cmd)
    ser.flush()
    print(f"Comando enviado: {cmd.decode().strip()}")


def list_ports():
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    if not ports:
        print("No se detectaron puertos seriales. Conecta el Grove Vision AI V2 por USB-C.")
        return
    print("Puertos disponibles:")
    for p in ports:
        print(f"  {p.device}  -  {p.description}")


def extract_latency_ms(line: str):
    """Intenta extraer la latencia de inferencia (ms) de una linea del log.
    Devuelve None si la linea no trae un valor reconocible."""
    line = line.strip()
    if not line:
        return None

    # Intento 1: JSON estandar SSCMA -> data.perf = [preprocess, inference, postprocess] en ms
    try:
        obj = json.loads(line)
        perf = obj.get("data", {}).get("perf")
        if isinstance(perf, list) and len(perf) >= 2:
            return float(perf[1])  # indice 1 = tiempo de inferencia
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # Intento 2: patrones de respaldo
    for pat in FALLBACK_PATTERNS:
        m = pat.search(line)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if groups:
                # si el patron de perf trae 3 numeros, el del medio suele ser inferencia
                return float(groups[1]) if len(groups) >= 2 else float(groups[0])
    return None


def discover(port: str, baudrate: int, seconds: int = 15):
    """Modo de descubrimiento: imprime las lineas crudas para confirmar el formato
    real del log antes de confiar en el parseo automatico."""
    print(f"Escuchando {port} @ {baudrate} baudios por {seconds}s. Apunta la camara a algo y observa.\n")
    with serial.Serial(port, baudrate, timeout=1) as ser:
        time.sleep(0.5)  # dar tiempo a que el puerto se estabilice
        start_continuous_invoke(ser)
        t0 = time.time()
        while time.time() - t0 < seconds:
            raw = ser.readline()
            if raw:
                try:
                    text = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    text = repr(raw)
                if text:
                    print(text)
    print("\nFin del modo descubrimiento. Copia un par de lineas de ejemplo "
          "y ajusta FALLBACK_PATTERNS / extract_latency_ms si el parseo automatico no funciono.")


def capture(port: str, baudrate: int, gamma: str, n: int, outfile: Path, raw_logfile: Path):
    """Captura N lecturas de latencia, descartando la primera (warmup)."""
    readings = []
    discarded_warmup = False

    with serial.Serial(port, baudrate, timeout=2) as ser, \
         open(raw_logfile, "a", encoding="utf-8") as rawlog:

        time.sleep(0.5)  # dar tiempo a que el puerto se estabilice
        start_continuous_invoke(ser)

        print(f"Capturando gamma={gamma} -> objetivo {n} lecturas validas (descarta warmup)...")
        while len(readings) < n:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue
            if not line:
                continue

            rawlog.write(line + "\n")

            lat = extract_latency_ms(line)
            if lat is None:
                continue

            if not discarded_warmup:
                discarded_warmup = True
                print(f"  [warmup descartado] {lat:.2f} ms")
                continue

            readings.append(lat)
            if len(readings) % 20 == 0:
                print(f"  ... {len(readings)}/{n}")

    median_lat = statistics.median(readings)
    mean_lat = statistics.mean(readings)
    stdev_lat = statistics.stdev(readings) if len(readings) > 1 else 0.0

    write_header = not outfile.exists()
    with open(outfile, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "gamma", "rep_index", "latency_ms"])
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for i, lat in enumerate(readings):
            writer.writerow([ts, gamma, i, lat])

    print(f"\nResultado gamma={gamma}:")
    print(f"  N={len(readings)}  mediana={median_lat:.2f} ms  media={mean_lat:.2f} ms  std={stdev_lat:.2f} ms")
    print(f"  Guardado en: {outfile}")
    print(f"  Log crudo en: {raw_logfile}")

    return median_lat


def main():
    ap = argparse.ArgumentParser(description="Captura de latencia Tier 2 - Grove Vision AI V2")
    ap.add_argument("--port", help="Puerto serial, ej. /dev/tty.usbmodemXXXX")
    ap.add_argument("--baudrate", type=int, default=BAUDRATE_DEFAULT)
    ap.add_argument("--n", type=int, default=100, help="Numero de inferencias validas a capturar")
    ap.add_argument("--gamma", default="NA", help="Etiqueta de la condicion (ej. 0.3, 0.6, 1.0...)")
    ap.add_argument("--out", default="results/tier2_latency.csv")
    ap.add_argument("--rawlog", default="results/tier2_raw_log.txt")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--discover", action="store_true",
                     help="Solo escucha e imprime el log crudo para confirmar el formato")
    args = ap.parse_args()

    if args.list_ports:
        list_ports()
        return

    if not args.port:
        sys.exit("Falta --port. Usa --list-ports para ver los disponibles.")

    if args.discover:
        discover(args.port, args.baudrate)
        return

    outfile = Path(args.out)
    rawlogfile = Path(args.rawlog)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    capture(args.port, args.baudrate, args.gamma, args.n, outfile, rawlogfile)


if __name__ == "__main__":
    main()