import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

# 1. Configuración principal de la página
st.set_page_config(
    page_title="CNN Clasificador - Los Benditos",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Caché para cargar los modelos de forma eficiente
@st.cache_resource
def cargar_modelos():
    # 1. Escribimos la arquitectura exacta que usaste en tu Colab
    def construir_cnn():
        modelo = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return modelo

    # 2. Construimos dos "cerebros" vacíos
    modelo_numeros = construir_cnn()
    modelo_ropa = construir_cnn()

    # 3. Le inyectamos SOLO el conocimiento (pesos), saltándonos la configuración rota
    # (Asegúrate de que los nombres coincidan con los que tienes en GitHub)
    modelo_numeros.load_weights('modelo_mnist.keras')
    modelo_ropa.load_weights('modelo_fashion.keras')
    
    return modelo_numeros, modelo_ropa

try:
    modelo_numeros, modelo_ropa = cargar_modelos()
except Exception as e:
    # Ahora sí, vamos a imprimir el error real que nos da el sistema
    st.error(f"⚠️ Error real al cargar los modelos: {str(e)}")
    st.stop()

# Diccionario de clases para Fashion MNIST
clases_ropa = ['Camiseta/Top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo',
               'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota de tobillo']

# 3. Función de preprocesamiento de imagen
def procesar_imagen(imagen_subida, invertir_colores=True):
    img = Image.open(imagen_subida).convert('L')
    if invertir_colores:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img, img_array

# 4. Diseño de la Barra Lateral (Sidebar)
st.sidebar.markdown("## 🛡️ Equipo: LOS BENDITOS")
st.sidebar.markdown("---")
st.sidebar.markdown("**Universidad Continental**")
st.sidebar.markdown("*Facultad de Ingeniería de Sistemas*")
st.sidebar.markdown("---")
st.sidebar.info(
    "Este proyecto aplica la metodología CRISP-ML para el despliegue "
    "de modelos de aprendizaje profundo (Deep Learning)."
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 - Cusco, Perú")

# 5. Estructura de Pestañas (Tabs)
tab1, tab2, tab3 = st.tabs(["🏠 Inicio y Contexto", "🧠 Clasificador Interactivo", "👥 El Equipo"])

# --- PESTAÑA 1: INICIO ---
with tab1:
    st.title("Clasificación de imágenes MNIST y Fashion-MNIST usando CNN")
    st.markdown("### Bienvenido al portal predictivo de Los Benditos")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            """
            En el contexto actual del avance acelerado de la Inteligencia Artificial, 
            el aprendizaje profundo se ha consolidado como una herramienta principal para 
            resolver problemas complejos de visión por computadora.
            
            Este proyecto aplica rigurosamente la metodología **CRISP-ML** (Cross-Industry Standard 
            Process for Machine Learning) para garantizar resultados confiables, reproducibles y 
            alineados con objetivos técnicos de calidad.
            """
        )
    with colB:
        st.success("**Objetivo Principal:**")
        st.markdown(
            """
            Conceptualizar, diseñar, implementar y evaluar modelos de Redes Neuronales 
            Convolucionales (CNN) orientados a la clasificación automática de imágenes, 
            comparando el desempeño frente a datasets de distinta complejidad visual.
            """
        )
    
    st.divider()
    st.metric(label="Rendimiento Esperado (Accuracy MNIST)", value="~98.0%", delta="Alto rendimiento")
    st.metric(label="Rendimiento Esperado (Accuracy Fashion-MNIST)", value="85% - 90%", delta="Complejidad Media-Alta")

# --- PESTAÑA 2: PREDICCIÓN ---
with tab2:
    st.header("Interfaz de Predicción en Tiempo Real")
    st.markdown("Selecciona el modelo que deseas probar y sube una imagen para su análisis.")
    
    # Selector de modelo integrado en la interfaz principal
    tipo_modelo = st.radio(
        "Seleccione el dataset objetivo:",
        ("Dígitos Manuscritos (MNIST)", "Prendas de Vestir (Fashion-MNIST)"),
        horizontal=True
    )
    
    if tipo_modelo == "Dígitos Manuscritos (MNIST)":
        st.subheader("🔢 Analizador numérico (0-9)")
        modelo_actual = modelo_numeros
        nombres_clases = [str(i) for i in range(10)]
        invertir_por_defecto = True
    else:
        st.subheader("👕 Analizador de prendas de vestir")
        modelo_actual = modelo_ropa
        nombres_clases = clases_ropa
        invertir_por_defecto = False

    archivo_subido = st.file_uploader("Sube una imagen (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

    with st.expander("⚙️ Opciones avanzadas de procesamiento de imagen"):
        invertir = st.checkbox(
            "Invertir colores (Necesario si el fondo es blanco y el objeto negro)", 
            value=invertir_por_defecto
        )

    if archivo_subido is not None:
        col1, col2 = st.columns([1, 2]) # Proporción de columnas
        
        with st.spinner('Procesando a través de la Red Neuronal Convolucional...'):
            img_mostrar, img_lista_para_modelo = procesar_imagen(archivo_subido, invertir_colores=invertir)
            
            predicciones = modelo_actual.predict(img_lista_para_modelo)[0]
            clase_predicha_idx = np.argmax(predicciones)
            clase_predicha_nombre = nombres_clases[clase_predicha_idx]
            confianza = predicciones[clase_predicha_idx] * 100

            with col1:
                st.markdown("### Imagen Procesada (28x28)")
                st.image(img_mostrar, use_container_width=True, clamp=True)
                
                st.success(f"**Resultado:** {clase_predicha_nombre}")
                st.info(f"**Nivel de Confianza:** {confianza:.2f}%")
                
                if confianza > 90:
                    st.balloons() # Efecto visual de celebración si la confianza es muy alta

            with col2:
                st.markdown("### Distribución de Probabilidades")
                df_probabilidades = pd.DataFrame({
                    'Clase': nombres_clases,
                    'Probabilidad (%)': predicciones * 100
                }).set_index('Clase')
                
                # Gráfico interactivo con color corporativo
                st.bar_chart(df_probabilidades, color="#1f77b4", height=350)
                
    else:
        st.warning("👈 Por favor, sube una imagen para iniciar la inferencia del modelo.")

# --- PESTAÑA 3: EQUIPO ---
with tab3:
    st.header("Conoce a LOS BENDITOS")
    st.markdown("Este proyecto fue desarrollado íntegramente por el siguiente equipo:")
    
    st.markdown(
        """
        * **Callañaupa Cjuiro, Manuel Fabrizio** (Participación: 100%)
        * **Jancco Salas, Juan Sebastián** (Participación: 100%)
        * **Quispe Cruz, Joel Milton** (Participación: 100%)
        * **Paredes Estrada, Marcelo Flavio** (Participación: 100%)
        * **Lerma Condori, Antoni** (Participación: 100%)
        """
    )
    
    st.divider()
    st.subheader("Datos Académicos")
    st.markdown("**Asignatura:** Construcción de Software")
    st.markdown("**Docente:** Ing. Hugo Espetia")
    st.markdown("**Sede:** Cusco - Perú, 2026")
