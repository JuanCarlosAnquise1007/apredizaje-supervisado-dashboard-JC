import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st

from modules.ui.theme import page_header

# ---------------------- UI UTILIDADES DE DATOS ----------------------

def load_sample_dataset():
    try:
        df = sns.load_dataset('mpg')
        # Ajuste de nombres más claros
        df = df.rename(columns={'mpg': 'consumo', 'desplazamiento': 'desplazamiento', 'acceleration': 'aceleracion'})
        return df
    except Exception:
        return pd.DataFrame()


def data_option():
    page_header('Opción 1: Carga y preparación de datos', 'Carga, limpieza y codificación de variables.', '📦')
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('Sube un archivo CSV/Excel o usa el dataset de ejemplo (rendimiento de automóviles).')

    uploaded = st.file_uploader('Subir CSV o Excel', type=['csv', 'xlsx'])
    df = st.session_state.get('df') 
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state['df'] = df
        except Exception as e:
            st.error(f'Error al leer archivo: {e}')
    else:
        if st.button('Usar dataset de ejemplo (automóviles)'):
            df = load_sample_dataset()
            st.session_state['df'] = df

    if df is None or df.empty:
        st.info('Aún no hay datos cargados.')
        return None

    st.subheader('Vista previa')
    st.dataframe(df)

    st.subheader('Información de columnas')
    info_df = pd.DataFrame({
        'tipo': df.dtypes.astype(str),
        'nulos': df.isna().sum()
    })
    st.dataframe(info_df)

    st.write('Total de valores faltantes por columna:')
    st.write(df.isna().sum())

    if st.button('Rellenar NaN numéricos con el promedio de su columna'):
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        st.success('Valores nulos numéricos rellenados con promedio.')

    st.write('Verificación de nulos después del relleno:')
    st.write(df.isna().sum())

    # Codificación de variables categóricas (One-Hot) simplificada
    st.subheader('Codificación de variables categóricas')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        drop_first = st.checkbox('Eliminar una categoría por variable para evitar colinealidad (drop_first)', value=True)
        if st.button('Convertir categóricas a numéricas (One-Hot)'):
            df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
            st.success('Conversión realizada.')
            st.dataframe(df)
            st.session_state['df'] = df
            # Permitir eliminar columnas dummy específicas para evitar colinealidad
            posibles_dummies = [c for c in df.columns if any(c.startswith(col + '_') for col in cat_cols)]
            if posibles_dummies:
                to_drop = st.multiselect('Selecciona columnas dummy a eliminar', options=posibles_dummies)
                if st.button('Eliminar columnas seleccionadas') and to_drop:
                    df = df.drop(columns=to_drop, errors='ignore')
                    st.success('Columnas dummy eliminadas.')
                    st.dataframe(df)
                    st.session_state['df'] = df
                # NUEVO: eliminar TODAS las columnas dummy generadas
                if st.button('Eliminar TODAS las columnas dummy generadas'):
                    df = df.drop(columns=posibles_dummies, errors='ignore')
                    st.success('Se eliminaron todas las columnas dummy.')
                    st.dataframe(df)
                    st.session_state['df'] = df
                # NUEVO: mostrar solo la columna "x11" en la vista
                if 'x11' in df.columns:
                    if st.button('Mostrar solo la columna "x11"'):
                        st.dataframe(df[['x11']])
                else:
                    st.info('La columna "x11" no existe en los datos actuales.')
    else:
        st.info('No se detectaron columnas categóricas.')

    st.session_state['df'] = df
    return df