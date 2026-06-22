# TFG oficial

Carpeta curada del Trabajo de Fin de Grado:

**Modelos de analisis supervisado para el analisis del Value at Risk de una cartera de activos financieros**.

Este repositorio contiene los datos, resultados y notebooks necesarios para reproducir tanto la comparacion final del TFG como el analisis de sensibilidad que justifica la arquitectura del modelo neuronal (Capitulo 5).

## Estructura

- `data/`: dataset base y, si se regenera desde cero, precios descargados.
- `results/predictions/`: predicciones finales de cada modelo.
- `results/tables/`: tablas por modelo, tablas anuales y comparacion final completa.
- `figures/`: figuras generadas para la memoria.
- `notebooks/`: notebooks ordenados y autosuficientes para generar datos, predicciones, tablas y graficas.
- `notebooks/sensitivity/`: notebooks de las pruebas de sensibilidad del modelo MLP-VaR.
- `results/sensitivity/`: predicciones y tablas de backtesting del analisis de sensibilidad.
- `mlp_experiments/`: las siete configuraciones MLP-VaR usadas para justificar la tabla de sensibilidad del Capitulo 5 (configuracion, notebooks, predicciones y tablas).

## Modelo MLP final

El modelo neuronal final incluido es:

`MLP Softplus + SiLU LayerNorm + downside pressure`

La memoria presenta el modelo final, justificado mediante el analisis de sensibilidad documentado en `mlp_experiments/`, y lo compara contra:

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

Los notebooks contienen el codigo necesario directamente en sus celdas. No dependen de una carpeta auxiliar ni de archivos situados fuera del repositorio.

Nota: el notebook `00_construccion_dataset.ipynb` necesita conexion a internet si se quiere reconstruir el dataset desde cero mediante `yfinance`. Para reproducir resultados desde los datos ya incluidos, se puede empezar desde los notebooks posteriores.

## Entorno

Crear un entorno de Python e instalar las dependencias minimas:

```bash
pip install -r requirements.txt
```

Los notebooks mas costosos son `02_entrenamiento_mlp_final.ipynb` y `06_benchmark_caviar_as.ipynb`, porque recalculan modelos rolling. Las tablas y figuras ya incluidas permiten revisar los resultados sin repetir esos entrenamientos.
