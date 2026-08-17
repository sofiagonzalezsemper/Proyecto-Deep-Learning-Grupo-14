"""
Asistente de Sentimiento Financiero (FinBERT) — prototipo Human in the Loop.

El modelo (FinBERT afinado en la Práctica 3 sobre Financial PhraseBank)
predice el sentimiento de una frase financiera y muestra, vía LIME, qué
palabras influyeron en esa predicción. La decisión final la toma una
persona: confirma o corrige la predicción antes de que quede registrada.

Ejecutar con:
    streamlit run app.py
"""

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from explain_utils import explicacion_a_lista, explicar_prediccion
from model_utils import (
    DEVICE,
    ID_TO_LABEL,
    LABEL_COLORS,
    cargar_modelo,
    modelo_fine_tuned_disponible,
    predecir_probabilidades,
)

st.set_page_config(
    page_title="Asistente de Sentimiento Financiero",
    page_icon="📊",
    layout="centered",
)

if "historial" not in st.session_state:
    st.session_state.historial = []  # lista de dicts, ver registrar_decision()


# --------------------------------------------------------------------------
# Utilidades de presentación
# --------------------------------------------------------------------------

def probabilidades_como_df(dict_probs):
    return (
        pd.DataFrame({"Sentimiento": list(dict_probs.keys()), "Probabilidad": list(dict_probs.values())})
        .set_index("Sentimiento")
    )


def resaltar_palabras_html(pares):
    """Arma un bloque HTML con las palabras más influyentes coloreadas:
    verde = a favor de la clase predicha, rojo = en contra."""
    max_peso = max((abs(peso) for _, peso in pares), default=1) or 1
    spans = []
    for palabra, peso in pares:
        intensidad = min(abs(peso) / max_peso, 1.0)
        color = "46,139,87" if peso > 0 else "192,57,43"  # verde / rojo (RGB)
        alpha = 0.15 + 0.55 * intensidad
        spans.append(
            f'<span style="background-color: rgba({color},{alpha}); '
            f'padding:2px 6px; margin:2px; border-radius:4px; display:inline-block;">'
            f"{palabra}</span>"
        )
    return " ".join(spans)


def registrar_decision(texto, etiqueta_modelo, confianza, decision_humana):
    st.session_state.historial.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "texto": texto,
            "prediccion_modelo": etiqueta_modelo,
            "confianza_modelo": round(confianza, 4),
            "decision_humana": decision_humana,
            "coincide": etiqueta_modelo == decision_humana,
        }
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Sobre este prototipo")
    st.write(
        "Arquetipo de producto: **Human in the Loop**. El modelo sugiere "
        "el sentimiento y lo explica; la persona confirma o corrige antes "
        "de que la lectura se considere válida."
    )

    with st.spinner("Cargando modelo…"):
        try:
            cargar_modelo()
            error_carga = None
        except Exception as exc:  # noqa: BLE001 — se muestra el error tal cual en la UI
            error_carga = str(exc)

    if error_carga:
        st.error(f"No se pudo cargar el modelo: {error_carga}")
    elif modelo_fine_tuned_disponible():
        st.success("Usando el modelo FinBERT afinado en la Práctica 3.")
    else:
        st.warning(
            "No se encontró el checkpoint de la Práctica 3 en "
            "`modelo/finbert_sentimiento_financiero_practica3/`. "
            "Usando FinBERT base (sin fine-tuning sobre Financial "
            "PhraseBank) — las predicciones funcionan pero no van a "
            "coincidir exactamente con las del informe."
        )

    st.caption(f"Dispositivo: {DEVICE}")

    num_samples_lime = st.slider(
        "Precisión de la explicación (LIME)",
        min_value=100,
        max_value=500,
        value=200,
        step=50,
        help="Más muestras = explicación más estable pero más lenta.",
    )

    if st.session_state.historial:
        df_historial = pd.DataFrame(st.session_state.historial)
        st.download_button(
            "Descargar historial (CSV)",
            data=df_historial.to_csv(index=False).encode("utf-8"),
            file_name="historial_decisiones.csv",
            mime="text/csv",
        )


st.title("📊 Asistente de Sentimiento Financiero")
st.caption("FinBERT + explicabilidad (LIME) · Human in the Loop")

tab_individual, tab_lote, tab_historial = st.tabs(
    ["Frase individual", "Lote (CSV)", "Historial de decisiones"]
)

# --------------------------------------------------------------------------
# Tab 1: frase individual
# --------------------------------------------------------------------------

with tab_individual:
    ejemplo = (
        "The company's profits did not increase as expected during the "
        "third quarter."
    )
    texto = st.text_area(
        "Frase o titular financiero (en inglés, idioma del dataset de entrenamiento)",
        value="",
        placeholder=ejemplo,
        height=100,
    )

    if st.button("Analizar", type="primary", disabled=not texto.strip()):
        with st.spinner("Prediciendo…"):
            probs = predecir_probabilidades([texto])[0]
            dict_probs = {ID_TO_LABEL[i]: float(probs[i]) for i in range(len(probs))}
            etiqueta_predicha = max(dict_probs, key=dict_probs.get)
            confianza = dict_probs[etiqueta_predicha]

        st.session_state["ultimo_texto"] = texto
        st.session_state["ultima_prediccion"] = etiqueta_predicha
        st.session_state["ultima_confianza"] = confianza
        st.session_state["ultimas_probs"] = dict_probs

    if st.session_state.get("ultimo_texto"):
        etiqueta_predicha = st.session_state["ultima_prediccion"]
        confianza = st.session_state["ultima_confianza"]
        dict_probs = st.session_state["ultimas_probs"]

        color = LABEL_COLORS.get(etiqueta_predicha, "#333333")
        st.markdown(
            f"### Predicción del modelo: "
            f'<span style="color:{color}">{etiqueta_predicha}</span> '
            f"({confianza:.1%} de confianza)",
            unsafe_allow_html=True,
        )
        st.bar_chart(probabilidades_como_df(dict_probs))

        with st.spinner("Calculando explicación (LIME)…"):
            explicacion = explicar_prediccion(
                st.session_state["ultimo_texto"], num_samples=num_samples_lime
            )
            pares = explicacion_a_lista(explicacion, etiqueta_predicha)

        st.markdown("**Palabras que más influyeron en esta predicción:**")
        st.markdown(resaltar_palabras_html(pares), unsafe_allow_html=True)
        st.caption("Verde = a favor de la clase predicha · Rojo = en contra.")

        st.divider()
        st.markdown("#### Revisión humana")
        st.write("¿Confirmás esta predicción o la corregís antes de registrarla?")

        opciones = list(ID_TO_LABEL.values())
        decision = st.radio(
            "Sentimiento final (según tu criterio)",
            opciones,
            index=opciones.index(etiqueta_predicha),
            horizontal=True,
            key="decision_radio",
        )

        if st.button("Guardar decisión"):
            registrar_decision(
                st.session_state["ultimo_texto"], etiqueta_predicha, confianza, decision
            )
            st.success("Decisión registrada en el historial.")


# --------------------------------------------------------------------------
# Tab 2: lote (CSV)
# --------------------------------------------------------------------------

with tab_lote:
    st.write(
        "Subí un CSV con una columna de texto para clasificar varias frases "
        "a la vez. Después podés revisar y corregir cada predicción antes "
        "de exportar los resultados."
    )
    archivo = st.file_uploader("Archivo CSV", type=["csv"])

    if archivo is not None:
        df_entrada = pd.read_csv(archivo)
        columna_texto = st.selectbox("Columna con el texto a analizar", df_entrada.columns)

        if st.button("Clasificar lote"):
            with st.spinner(f"Prediciendo {len(df_entrada)} frases…"):
                textos = df_entrada[columna_texto].astype(str).tolist()
                probs = predecir_probabilidades(textos)
                predicciones = [ID_TO_LABEL[int(p.argmax())] for p in probs]
                confianzas = [float(p.max()) for p in probs]

            df_resultado = df_entrada.copy()
            df_resultado["prediccion_modelo"] = predicciones
            df_resultado["confianza_modelo"] = [round(c, 4) for c in confianzas]
            df_resultado["decision_humana"] = predicciones  # se corrige abajo

            st.session_state["df_lote"] = df_resultado

        if "df_lote" in st.session_state:
            st.write("Corregí la columna **decision_humana** donde no estés de acuerdo con el modelo:")
            df_editado = st.data_editor(
                st.session_state["df_lote"],
                column_config={
                    "decision_humana": st.column_config.SelectboxColumn(
                        "decision_humana", options=list(ID_TO_LABEL.values())
                    )
                },
                disabled=[columna_texto, "prediccion_modelo", "confianza_modelo"],
                use_container_width=True,
                key="editor_lote",
            )

            csv_buffer = io.StringIO()
            df_editado.to_csv(csv_buffer, index=False)
            st.download_button(
                "Descargar resultados revisados (CSV)",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="predicciones_revisadas.csv",
                mime="text/csv",
            )

            if st.button("Agregar lote al historial"):
                for _, fila in df_editado.iterrows():
                    registrar_decision(
                        fila[columna_texto],
                        fila["prediccion_modelo"],
                        fila["confianza_modelo"],
                        fila["decision_humana"],
                    )
                st.success(f"Se agregaron {len(df_editado)} decisiones al historial.")


# --------------------------------------------------------------------------
# Tab 3: historial
# --------------------------------------------------------------------------

with tab_historial:
    if not st.session_state.historial:
        st.info("Todavía no hay decisiones registradas en esta sesión.")
    else:
        df_historial = pd.DataFrame(st.session_state.historial)
        st.dataframe(df_historial, use_container_width=True)

        total = len(df_historial)
        coincidencias = int(df_historial["coincide"].sum())
        st.metric(
            "Coincidencia modelo ↔ humano",
            f"{coincidencias}/{total} ({coincidencias / total:.0%})",
        )
