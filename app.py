# ============================================================
# App de Streamlit: Riesgo actuarial de clientes de seguros médicos
# Modelos: K-Means (3 clusters) + PCA (visualización) + SVM (apoyo)
# La persona ingresa los datos de un cliente y la app indica el
# cluster, el nivel de riesgo (Bajo, Medio o Alto) y una explicación.
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuración general de la página
# ------------------------------------------------------------
st.set_page_config(page_title="Riesgo actuarial", page_icon="🌸", layout="centered")

# Colores de la app (rosa viejo / rosa pastel)
ROSA_FONDO = "#f6dfe4"       # fondo rosa pastel
ROSA_TARJETA = "#fbeef1"     # tarjetas
ROSA_VIEJO = "#a8566b"       # acentos (rosa viejo)
TEXTO = "#3b2630"            # texto oscuro para que se lea bien
COLORES_RIESGO = {"Bajo": "#7fb77e", "Medio": "#e0a43a", "Alto": "#c0504d"}

# Estilo: fondo rosa y texto oscuro (también por si el navegador está en modo oscuro)
st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, {ROSA_FONDO} 0%, #efcfd7 100%); }}
    .stApp, .stApp p, .stApp label, .stApp li, .stApp span,
    .stApp h1, .stApp h2, .stApp h3, .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricLabel"], .stApp [data-testid="stCaptionContainer"] {{
        color: {TEXTO};
    }}
    h1, h2, h3 {{ color: {ROSA_VIEJO} !important; }}
    [data-testid="stForm"] {{
        background-color: {ROSA_TARJETA};
        border: 1px solid #d9a8b4;
        border-radius: 14px;
        padding: 18px;
    }}
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{
        background-color: #ffffff;
        color: {TEXTO};
    }}
    .stButton > button, [data-testid="stFormSubmitButton"] > button {{
        background-color: {ROSA_VIEJO};
        color: #ffffff;
        border: none;
        border-radius: 10px;
    }}
    [data-testid="stMetric"] {{
        background-color: {ROSA_TARJETA};
        border-radius: 12px;
        padding: 10px 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Predicción de riesgo actuarial - Vianey Diaz - ISC - Campus Comayagua")


# ------------------------------------------------------------
# Función auxiliar: busca un archivo en la raíz o en las carpetas
# models/ y outputs/ (así funciona sin importar dónde lo subas)
# ------------------------------------------------------------
def buscar_archivo(nombre):
    rutas = [nombre, f"models/{nombre}", f"outputs/{nombre}"]
    for ruta in rutas:
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(f"No se encontró el archivo: {nombre}")


# ------------------------------------------------------------
# Carga de modelos y metadata (una sola vez gracias al caché)
# ------------------------------------------------------------
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load(buscar_archivo("kmeans_riesgo_actuarial.pkl"))  # preparación + K-Means
    pca = joblib.load(buscar_archivo("pca_riesgo.pkl"))                  # PCA de 2 componentes
    svm = joblib.load(buscar_archivo("svm_riesgo.pkl"))                  # SVM de apoyo
    with open(buscar_archivo("model_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, pca, svm, metadata


# ------------------------------------------------------------
# Carga de la base de datos y cálculo de la vista PCA de todos los clientes
# ------------------------------------------------------------
@st.cache_data
def cargar_base():
    return pd.read_csv(buscar_archivo("insurance.csv")).drop_duplicates().reset_index(drop=True)


modelo, pca, svm, metadata = cargar_modelos()
df = cargar_base()

# Datos guardados en la metadata desde Colab
variables = metadata["variables_numericas"] + metadata["variables_categoricas"]
mapa = {int(k): v for k, v in metadata["mapa_riesgo"].items()}     # cluster -> riesgo
perfiles = pd.DataFrame(metadata["perfiles"]).set_index("cluster")  # resumen por cluster
preprocesamiento = modelo.named_steps["preprocesamiento"]

st.caption(metadata["nombre_modelo"])

# Aviso si la versión de scikit-learn no coincide con la de Colab
if sklearn.__version__ != metadata.get("version_sklearn"):
    st.warning(
        f"Los modelos se entrenaron con scikit-learn {metadata.get('version_sklearn')} "
        f"pero la app usa {sklearn.__version__}. Ajuste requirements.txt."
    )

# Clusters y coordenadas PCA de toda la base (para la gráfica)
X_base = df[variables]
df["cluster"] = modelo.predict(X_base)
df["riesgo"] = df["cluster"].map(mapa)
coords_base = pca.transform(preprocesamiento.transform(X_base))

# ------------------------------------------------------------
# Formulario: la persona ingresa los datos del cliente
# ------------------------------------------------------------
st.subheader("Datos del cliente")

with st.form("datos"):
    col1, col2 = st.columns(2)

    age = col1.number_input("Edad", 18, 100, 35)
    sex = col2.selectbox("Sexo", sorted(df["sex"].unique()))

    bmi = col1.number_input("BMI", 10.0, 60.0, 28.0)
    children = col2.number_input("Hijos", 0, 10, 1)

    smoker = col1.selectbox("Fumador", sorted(df["smoker"].unique()))
    region = col2.selectbox("Región", sorted(df["region"].unique()))

    charges = st.number_input("Cargos médicos (charges)", 0.0, 100000.0, 12000.0)

    enviar = st.form_submit_button("Evaluar")

# ------------------------------------------------------------
# Resultado (solo cuando se presiona "Evaluar")
# ------------------------------------------------------------
if enviar:
    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, "children": children,
        "smoker": smoker, "region": region, "charges": charges,
    }])

    # Cluster asignado por K-Means y nivel de riesgo
    cluster = int(modelo.predict(cliente[variables])[0])
    riesgo = mapa.get(cluster, "No definido")
    perfil = perfiles.loc[cluster]

    st.subheader("Resultado")
    mensaje = f"Riesgo actuarial: {riesgo}"
    if riesgo == "Bajo":
        st.success(mensaje)
    elif riesgo == "Medio":
        st.warning(mensaje)
    else:
        st.error(mensaje)

    c1, c2 = st.columns(2)
    c1.metric("Cluster asignado", cluster)
    c2.metric("Clientes en este cluster", int(perfil["clientes"]))

    # --------------------------------------------------------
    # Explicación breve: compara al cliente con el promedio de su cluster
    # --------------------------------------------------------
    es_fumador = smoker == "yes"
    frases = [
        f"El cliente fue asignado al cluster {cluster}, que corresponde a riesgo {riesgo.lower()}."
    ]
    frases.append(
        f"En este grupo el cargo médico promedio es de ${perfil['cargo_promedio']:,.0f} "
        f"y el cliente tiene ${charges:,.0f}."
    )
    frases.append(
        f"La edad promedio del grupo es {perfil['edad_promedio']:.0f} años "
        f"(el cliente tiene {age}) y el BMI promedio es {perfil['bmi_promedio']:.1f} "
        f"(el cliente tiene {bmi:.1f})."
    )
    if es_fumador:
        frases.append(
            f"El cliente es fumador, y en este grupo el {perfil['pct_fumadores']:.0f}% "
            "de las personas fuma."
        )
    else:
        frases.append(
            f"El cliente no fuma, y en este grupo el {100 - perfil['pct_fumadores']:.0f}% "
            "de las personas no fuma."
        )
    st.info(" ".join(frases))

    # El SVM de apoyo da su propia opinión sobre el cluster
    cluster_svm = int(svm.predict(preprocesamiento.transform(cliente[variables]))[0])
    if cluster_svm == cluster:
        st.caption("El SVM de apoyo coincide con la asignación de K-Means.")
    else:
        st.caption(
            f"El SVM de apoyo ubica al cliente en el cluster {cluster_svm} "
            f"(riesgo {mapa.get(cluster_svm, 'No definido').lower()}): el cliente está cerca "
            "de la frontera entre dos grupos."
        )

    # --------------------------------------------------------
    # Visualización: clientes en el plano PCA y el cliente evaluado
    # --------------------------------------------------------
    st.write("Ubicación del cliente entre los clusters (PCA)")
    punto = pca.transform(preprocesamiento.transform(cliente[variables]))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(ROSA_TARJETA)
    ax.set_facecolor("#ffffff")
    for nivel, color in COLORES_RIESGO.items():
        marca = (df["riesgo"] == nivel).values
        ax.scatter(coords_base[marca, 0], coords_base[marca, 1], s=14, alpha=0.65,
                   color=color, label=f"Riesgo {nivel}")
    ax.scatter(punto[:, 0], punto[:, 1], marker="*", s=420, color="black",
               edgecolors="white", linewidths=1.3, label="Cliente evaluado", zorder=5)
    ax.set(xlabel=f"PC1 ({metadata['varianza_pca'][0] * 100:.1f}%)",
           ylabel=f"PC2 ({metadata['varianza_pca'][1] * 100:.1f}%)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------
# Resumen del análisis (viene de la metadata y la base)
# ------------------------------------------------------------
st.divider()
st.subheader("Resumen de los clusters")

c1, c2, c3 = st.columns(3)
c1.metric("Clusters (k)", metadata["k"])
c2.metric("Silhouette", f"{metadata['silhouette']:.2f}")
c3.metric("Accuracy SVM", f"{metadata['svm_accuracy'] * 100:.1f}%")

tabla = perfiles.copy()
tabla["Riesgo"] = tabla.index.map(mapa)
tabla = tabla.reset_index().set_index("Riesgo").loc[["Bajo", "Medio", "Alto"]]
tabla = tabla.rename(columns={
    "cluster": "Cluster", "clientes": "Clientes", "edad_promedio": "Edad prom.",
    "bmi_promedio": "BMI prom.", "hijos_promedio": "Hijos prom.",
    "cargo_promedio": "Cargo prom.", "pct_fumadores": "% fumadores",
}).drop(columns=["riesgo"], errors="ignore")
st.dataframe(tabla, width="stretch")

# Gráfica del método del codo y silhouette (guardados desde Colab)
st.write("Método del codo y silhouette score")
codo = pd.DataFrame(metadata["tabla_codo_silhouette"])
fig2, ejes = plt.subplots(1, 2, figsize=(9, 3.4))
fig2.patch.set_facecolor(ROSA_TARJETA)
for ax in ejes:
    ax.set_facecolor("#ffffff")
ejes[0].plot(codo["k"], codo["inercia"], marker="o", color=ROSA_VIEJO)
ejes[0].set(title="Codo", xlabel="k", ylabel="Inercia")
ejes[1].plot(codo["k"], codo["silhouette"], marker="o", color=ROSA_VIEJO)
ejes[1].set(title="Silhouette", xlabel="k", ylabel="Silhouette")
plt.tight_layout()
st.pyplot(fig2)

st.write("Vista rápida de la base de datos")
st.dataframe(df.drop(columns=["cluster", "riesgo"]).head(20), width="stretch")
