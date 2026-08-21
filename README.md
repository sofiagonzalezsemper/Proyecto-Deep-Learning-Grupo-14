# Asistente de Sentimiento Financiero

## FinBERT + Human-in-the-Loop

Prototipo académico que clasifica frases financieras como positivas, negativas o neutrales y mantiene a una persona dentro del proceso de decisión. El sistema combina transfer learning con FinBERT, confianza de predicción y explicaciones locales mediante LIME.

## Equipo

- Santiago Vero
- Sofía González
- Malena dos Santos

## Problema

Los equipos de análisis necesitan procesar grandes volúmenes de noticias financieras. Una clasificación automática puede acelerar el trabajo, pero una predicción incorrecta no debería utilizarse sin revisión en un dominio sensible.

El flujo propuesto es:

`Texto → predicción → confianza → explicación LIME → confirmación/corrección humana → historial`

## Dataset

Se utilizó Financial PhraseBank:

| Característica | Valor |
|---|---:|
| Registros originales | 2.264 |
| Registros sin duplicados | 2.259 |
| Entrenamiento / validación / prueba | 70% / 15% / 15% |
| Neutral | 61% |
| Positivo | 25% |
| Negativo | 13% |

La partición fue estratificada y se aplicaron pesos de clase.

## Modelos comparados

| Modelo | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|
| MLP + TF-IDF | 87,32% | 83,35% | 87,40% |
| Transformer desde cero | 85,25% | 79,87% | 85,03% |
| FinBERT | **99,12%** | **98,62%** | **99,12%** |

FinBERT obtuvo el mejor resultado al aprovechar representaciones preentrenadas sobre lenguaje financiero. Dado el tamaño reducido del dataset y el resultado excepcionalmente alto, las métricas deben interpretarse dentro de esta partición académica y validarse sobre datos externos antes de afirmar capacidad de generalización en producción.

## Prototipo

La interfaz de Streamlit contempla:

- Predicción individual.
- Probabilidades por clase.
- Explicación de palabras mediante LIME.
- Confirmación o corrección humana.
- Clasificación de lotes CSV.
- Historial exportable.

## Estado actual del repositorio

Actualmente están publicados:

- [`app.py`](./app.py): capa de interfaz Streamlit.
- Este README con el diseño, metodología y resultados.

La aplicación importa `model_utils.py` y `explain_utils.py`, que todavía no están publicados. Tampoco se incluyen actualmente los notebooks, el checkpoint ni el archivo de dependencias. Por ese motivo, **el repositorio todavía no es ejecutable de forma independiente**.

Archivos pendientes para completar la reproducción:

```text
.
├── README.md
├── app.py
├── model_utils.py
├── explain_utils.py
├── requirements.txt
├── notebooks/
└── modelo/ o instrucciones de descarga
```

## Configuración experimental

| Parámetro | Valor |
|---|---|
| Modelo base | `ProsusAI/finbert` |
| Learning rate | `2e-5` |
| Optimizador | AdamW |
| Épocas máximas | 8 |
| Early stopping | Sí |
| Weight decay | Sí |
| Pérdida ponderada | Sí |
| Longitud máxima | 58 |

## Explicabilidad y revisión humana

LIME perturba palabras de una frase y observa los cambios en la predicción. La explicación permite revisar si el modelo se apoya en términos razonables, pero no demuestra causalidad ni garantiza que la decisión sea correcta.

Las correcciones humanas podrían utilizarse en el futuro para análisis de errores, detección de drift, active learning y nuevas iteraciones de entrenamiento.

## Consideraciones para producción

- Validación con noticias externas y más recientes.
- Pruebas específicas de negaciones y casos ambiguos.
- Versionado de dataset, modelo y tokenizer.
- Persistencia y auditoría del historial.
- Monitoreo por clase, confianza y drift.
- Separación entre interfaz, inferencia y explicabilidad.

## Licencia

Proyecto académico. Antes de reutilizar código, datos o checkpoints debe verificarse la licencia de cada dependencia y dataset.
