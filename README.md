Asistente de Sentimiento Financiero (FinBERT + Human-in-the-Loop)
Proyecto grupal del curso de Deep Learning: un asistente que clasifica el sentimiento (positivo, negativo, neutral) de noticias y frases financieras, combinando un modelo de Deep Learning de última generación con supervisión humana antes de que cualquier predicción se use para tomar decisiones.
Equipo: Santiago Vero, Sofía González y un tercer integrante.
Índice
Problema y motivación
Arquetipo de producto ML
Dataset
Ciclo de vida del proyecto
Modelos evaluados
Arquitectura final: FinBERT
Explicabilidad (LIME)
Prototipo (Streamlit)
Estructura del repositorio
Instalación y uso
Desafíos y aprendizajes
Consideraciones de producción
Video de presentación
Licencia
Problema y motivación
Analistas financieros y equipos de research necesitan leer y clasificar constantemente noticias y reportes para tomar decisiones de inversión: ¿esta frase indica que la empresa va bien, mal, o es neutral? Hacerlo manualmente a gran escala es lento y no escala; confiar ciegamente en un modelo automático sin supervisión es riesgoso, porque un error de clasificación puede derivar en una mala decisión financiera.
Antes de optar por Deep Learning evaluamos alternativas más simples: reglas basadas en diccionarios de sentimiento financiero y modelos clásicos de ML sobre features de texto (TF-IDF + MLP). Funcionan razonablemente bien, pero el lenguaje financiero tiene matices y negaciones ("las ganancias no aumentaron" vs. "las ganancias aumentaron") que una bolsa de palabras no captura. Por eso una arquitectura que entienda contexto —Deep Learning, y en particular Transformers— es la solución adecuada.
En términos de impacto vs. costo, el proyecto es de alto impacto (ahorra tiempo de análisis y estandariza criterios) y bajo costo de construcción, ya que parte de un dataset público (Financial PhraseBank) y de un modelo preentrenado especializado en finanzas (FinBERT) en vez de entrenar algo desde cero.
Arquetipo de producto ML
Human in the Loop. El modelo predice el sentimiento y explica su predicción (vía LIME), pero es un analista humano quien confirma o corrige esa predicción antes de que se use para tomar una decisión. Se eligió este arquetipo, y no un sistema autónomo, porque en el dominio financiero un error de clasificación tiene consecuencias reales y el modelo puede equivocarse en casos ambiguos: la IA acelera el trabajo del analista, no lo reemplaza sin supervisión.
Dataset
Financial PhraseBank: 2.264 frases de noticias financieras en inglés, etiquetadas como Negativo, Neutral o Positivo.
Tras eliminar duplicados quedaron 2.259 registros.
Desbalance de clases importante: 61% Neutral, 25% Positivo, 13% Negativo.
Partición estratificada 70/15/15 (train/validation/test) y pesos de clase durante el entrenamiento para que el modelo no ignore las clases minoritarias.
Ciclo de vida del proyecto
Planificación: se reutilizó el problema trabajado en prácticas individuales anteriores (clasificación de sentimiento financiero), aprovechando el dataset ya validado y los aprendizajes de dos iteraciones previas. Se definió desde el día uno el arquetipo Human-in-the-Loop y el prototipo en Streamlit, y se repartieron roles en el equipo.
Recolección y preparación de datos: carga y limpieza del Financial PhraseBank, análisis exploratorio, deduplicación y partición estratificada (ver Dataset).
Exploración y refinamiento del modelo: iteración de tres arquitecturas, de menor a mayor complejidad/adecuación, comparadas bajo las mismas condiciones (ver Modelos evaluados).
Evaluación: accuracy, F1 macro y F1 weighted sobre el mismo conjunto de prueba y la misma semilla en cada iteración, para que las comparaciones fueran justas.
Modelos evaluados
Métrica	MLP + TF-IDF	Transformer desde cero	FinBERT (transfer learning)
Accuracy	87,32%	85,25%	99,12%
F1 Macro	83,35%	79,87%	98,62%
F1 Weighted	87,40%	85,03%	99,12%
MLP + TF-IDF: baseline clásico, sirvió de piso de comparación.
Transformer entrenado desde cero: no logró superar al baseline simple; se sobreajustó porque con ~1.580 frases de entrenamiento no hay datos suficientes para que un Transformer aprenda sus propios embeddings desde cero.
FinBERT vía transfer learning: partir de un modelo que ya conoce el lenguaje financiero y ajustarlo con el dato específico disponible superó ampliamente a las otras dos opciones, confirmando que con un dataset chico, transfer learning desde un modelo especializado en el dominio es mucho más efectivo que entrenar una arquitectura compleja desde cero.
Arquitectura final: FinBERT
Se eligió ProsusAI/finbert, un modelo basado en BERT (bloques Transformer encoder con self-attention bidireccional) preentrenado específicamente sobre lenguaje financiero, con fine-tuning supervisado completo sobre el dataset del proyecto:
Learning rate: 2e-5
Optimizador: AdamW
8 épocas con early stopping
Weight decay
Pérdida ponderada por clase para compensar el desbalance
MAX_LENGTH = 58
Mapeo de clases: 0 = Positivo, 1 = Negativo, 2 = Neutral
Explicabilidad (LIME)
Para que el flujo Human-in-the-Loop tenga sentido, el modelo no solo predice: también se explica. Se usa LIME, que genera explicaciones locales perturbando palabras de la frase y observando cómo cambia la predicción, identificando qué palabras empujaron al modelo hacia cada clase. Esta información es clave para que el analista humano pueda confiar (o desconfiar) de una predicción puntual, en vez de aceptarla a ciegas.
Prototipo (Streamlit)
Flujo: input de texto → predicción + confianza → resaltado LIME → confirmación/corrección humana → historial exportable (CSV).
La app tiene tres pestañas:
Frase individual: se ingresa una frase, el modelo predice la clase con su nivel de confianza y se muestra el resaltado de LIME sobre las palabras más influyentes. El analista puede confirmar o corregir la predicción ahí mismo.
Lote (CSV): se sube un CSV con varias frases y el modelo las clasifica todas de una vez, para casos de mayor volumen.
Historial de decisiones: queda registrado todo lo que el analista fue confirmando o corrigiendo, exportable a CSV — datos que a futuro podrían usarse para reentrenar el modelo.
Estructura del repositorio
.
├── notebooks/
│   ├── practica1_deep_learning.ipynb        # Baseline: MLP + TF-IDF
│   ├── practica2_transformer.ipynb           # Transformer entrenado desde cero
│   └── practica3_finbert_transfer_learning.ipynb  # Fine-tuning de FinBERT + explicabilidad
├── prototipo/
│   ├── app.py            # Interfaz Streamlit (3 pestañas)
│   ├── model_utils.py    # Carga de FinBERT y función de predicción
│   ├── explain_utils.py  # Explicabilidad con LIME
│   ├── modelo/            # Checkpoint entrenado (finbert_sentimiento_financiero_practica3/)
│   ├── ejemplos.csv      # Frases de ejemplo para el modo lote
│   ├── requirements.txt
│   └── README.md
└── README.md              # Este archivo
Ajustá los nombres de carpetas/archivos de esta sección para que coincidan exactamente con los del repositorio antes de publicarlo.
Instalación y uso
Notebooks
pip install -r requirements.txt
jupyter notebook
Abrí el notebook correspondiente a cada práctica para reproducir el entrenamiento y la evaluación de cada modelo.
Prototipo Streamlit
cd prototipo
pip install -r requirements.txt
Copiá el checkpoint de FinBERT entrenado en la Práctica 3 (generado con trainer.save_model + tokenizer.save_pretrained) dentro de modelo/finbert_sentimiento_financiero_practica3/. Sin ese checkpoint, la app cae automáticamente al FinBERT base descargado de Hugging Face (requiere conexión a internet la primera vez).
streamlit run app.py
Desafíos y aprendizajes
El Transformer entrenado desde cero no superó al baseline simple por falta de datos, lo que obligó a pivotar hacia transfer learning.
Hubo que resolver con cuidado el mapeo de etiquetas entre el dataset propio y el orden interno de clases de FinBERT, que no coincide por defecto.
El prototipo se desarrolló y probó end-to-end con un modelo de reemplazo (dummy) en un entorno sin acceso directo a Hugging Face Hub, dejando documentado el paso de reemplazar ese checkpoint por el real antes de una demo en producción.
Aprendizaje clave: más arquitectura no es sinónimo de mejor resultado si no hay datos suficientes para respaldarla, y la explicabilidad no es un accesorio sino una pieza central cuando el arquetipo del producto depende de la confianza humana en el modelo.
Consideraciones de producción
Pensando en un despliegue real más allá del contexto de curso:
Monitoreo continuo del modelo para detectar drift, ya que el lenguaje financiero y los temas de las noticias cambian con el tiempo.
Deuda técnica: hoy se depende de descargar el modelo base desde Hugging Face Hub; en producción convendría un artefacto versionado propio.
Escalabilidad: si el volumen de frases a clasificar crece, procesamiento en lote e infraestructura de inferencia dedicada.

Licencia
[Agregar licencia, por ejemplo MIT]
 
