import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import seaborn as sns

from modules.utils.helpers import fig_to_bytes


def tree_option():
    st.header('Opción 4: Árbol de Decisión')
    df = st.session_state.get('df')
    if df is None or df.empty:
        st.warning('Primero carga/prepara datos en Opción 1.')
        return

    target = st.selectbox('Variable objetivo', options=df.columns.tolist())
    features = st.multiselect('Características (X)', options=[c for c in df.columns if c != target])
    test_size = st.slider('Proporción de prueba', 0.1, 0.5, 0.2, 0.05)

    if not target or not features:
        st.info('Selecciona objetivo y características.')
        return

    X = df[features]
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    is_regression = pd.api.types.is_numeric_dtype(y)

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=42)

    max_depth = st.slider('Profundidad máxima', 1, 20, 5)
    criterion = st.selectbox('Criterio', options=['gini', 'entropy'])
    ccp_alpha = st.slider('Pruning (ccp_alpha)', 0.0, 0.05, 0.0, 0.001)

    if is_regression:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42, ccp_alpha=ccp_alpha)
    else:
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42, ccp_alpha=ccp_alpha)

    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    st.subheader('Ganancia de información (Información mutua)')
    images = []
    interpretations = []
    try:
        if is_regression:
            mi = mutual_info_regression(X_train, y_train)
        else:
            mi = mutual_info_classif(X_train, y_train)
        mi_df = pd.DataFrame({'feature': X.columns, 'info_mutua': mi}).sort_values('info_mutua', ascending=False)
        st.dataframe(mi_df)
        fig_mi, ax_mi = plt.subplots(figsize=(8, 4))
        ax_mi.bar(mi_df['feature'], mi_df['info_mutua'])
        ax_mi.set_title('Información mutua por característica')
        ax_mi.tick_params(axis='x', rotation=45)
        st.pyplot(fig_mi)
        images.append({'titulo': 'Información mutua', 'bytes': fig_to_bytes(fig_mi)})
        interpretations.append('Las características con mayor información mutua aportan más a la predicción/decisión del árbol.')
    except Exception:
        st.info('No se pudo calcular la información mutua para estos datos.')

    if is_regression:
        r2_tr = r2_score(y_train, y_pred_train)
        r2_te = r2_score(y_test, y_pred_test)
        st.write(f'R² entrenamiento: {r2_tr:.3f} | R² prueba: {r2_te:.3f}')
        interpretations.append(f"El desempeño del árbol de regresión muestra R²_train={r2_tr:.3f} y R²_test={r2_te:.3f}. Una gran brecha sugiere sobreajuste; ccp_alpha ayuda a podar el árbol.")
    else:
        acc_tr = accuracy_score(y_train, y_pred_train)
        acc_te = accuracy_score(y_test, y_pred_test)
        st.write(f'Exactitud entrenamiento: {acc_tr:.3f} | Exactitud prueba: {acc_te:.3f}')
        interpretations.append(f"El árbol de clasificación muestra Acc_train={acc_tr:.3f} y Acc_test={acc_te:.3f}. Si Acc_train >> Acc_test, existe sobreajuste; ajuste max_depth/ccp_alpha puede mitigarlo.")

    st.subheader('Importancia de variables')
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    ax_imp.bar(importances.index, importances.values)
    ax_imp.set_title('Importancia de características')
    ax_imp.tick_params(axis='x', rotation=45)
    st.pyplot(fig_imp)
    images.append({'titulo': 'Importancia de características', 'bytes': fig_to_bytes(fig_imp)})

    st.subheader('Visualización del árbol')
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model, feature_names=X.columns, filled=True, max_depth=max_depth, class_names=[str(c) for c in np.unique(y)] if not is_regression else None)
    st.pyplot(fig)
    images.append({'titulo': 'Árbol de decisión', 'bytes': fig_to_bytes(fig)})

    if is_regression:
        st.session_state.setdefault('reports', []).append({
            'seccion': 'Árbol (Regresión)',
            'items': [{'R2_train': r2_tr, 'R2_test': r2_te, 'max_depth': max_depth}],
            'imagenes': images,
            'interpretaciones': interpretations,
            'fecha': datetime.now().isoformat()
        })
    else:
        st.session_state.setdefault('reports', []).append({
            'seccion': 'Árbol (Clasificación)',
            'items': [{'Acc_train': acc_tr, 'Acc_test': acc_te, 'max_depth': max_depth}],
            'imagenes': images,
            'interpretaciones': interpretations,
            'fecha': datetime.now().isoformat()
        })