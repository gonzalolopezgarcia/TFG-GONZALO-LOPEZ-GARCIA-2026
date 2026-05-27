# TFG oficial

Carpeta curada del Trabajo de Fin de Grado:

**Modelos de analisis supervisado para el analisis del Value at Risk de una cartera de activos financieros**.

Esta version contiene solo los datos, resultados y notebooks necesarios para reproducir la comparacion final del TFG. Los notebooks experimentales y pruebas descartadas quedan fuera de esta carpeta.

## Estructura

- `data/`: dataset base y, si se regenera desde cero, precios descargados.
- `results/predictions/`: predicciones finales de cada modelo.
- `results/tables/`: tablas por modelo, tablas anuales y comparacion final completa.
- `figures/`: figuras generadas para la memoria.
- `notebooks/`: notebooks ordenados y autosuficientes para generar datos, predicciones, tablas y graficas.

## Modelo MLP final

El modelo neuronal final incluido es:

`MLP Softplus + SiLU LayerNorm + downside pressure`

No se incluye el MLP original experimental. La memoria presenta directamente el modelo final y lo compara contra:

- GARCH-t
- GARCH normal
- Simulacion historica
- CAViaR-AS

## Flujo recomendado

Ejecutar los notebooks en este orden:

1. `notebooks/00_construccion_dataset.ipynb`
2. `notebooks/01_datos_y_descriptivos.ipynb`
3. `notebooks/02_entrenamiento_mlp_final.ipynb`
4. `notebooks/03_benchmark_hs.ipynb`
5. `notebooks/04_benchmark_garch_t.ipynb`
6. `notebooks/05_benchmark_garch_normal.ipynb`
7. `notebooks/06_benchmark_caviar_as.ipynb`
8. `notebooks/07_comparacion_final.ipynb`
9. `notebooks/08_graficas_finales.ipynb`

Los notebooks contienen el codigo necesario directamente en sus celdas. No dependen de una carpeta auxiliar ni de archivos situados fuera de `TFG_oficial/`.

Nota: el notebook `00_construccion_dataset.ipynb` necesita conexion a internet si se quiere reconstruir el dataset desde cero mediante `yfinance`. Para reproducir resultados desde los datos ya incluidos, se puede empezar desde los notebooks posteriores.

## Entorno

Crear un entorno de Python e instalar las dependencias minimas:

```bash
pip install -r requirements.txt
```

Los notebooks mas costosos son `02_entrenamiento_mlp_final.ipynb` y `06_benchmark_caviar_as.ipynb`, porque recalculan modelos rolling. Las tablas y figuras ya incluidas permiten revisar los resultados sin repetir esos entrenamientos.
