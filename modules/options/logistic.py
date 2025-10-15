import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd

from modules.utils.helpers import fig_to_bytes, extract_coefficients


def logistic_option():
    st.header('Opción 3: Regresión Logística y ROC')
    df = st.session_state.get('df')
    if df is None or df.empty:
        st.warning('Primero carga/prepara datos en Opción 1.')
        return

    target = st.selectbox('Variable objetivo (categórica/binaria)', options=df.columns.tolist())
    features = st.multiselect('Características (X)', options=[c for c in df.columns if c != target])
    test_size = st.slider('Proporción de prueba', 0.1, 0.5, 0.2, 0.05)
    scale = st.checkbox('Estandarizar características')

    if not target or not features:
        st.info('Selecciona objetivo y características.')
        return

    y_raw = df[target]
    classes = sorted(y_raw.dropna().unique().tolist())
    if len(classes) == 2:
        pos_label = classes[1]
    else:
        pos_label = st.selectbox('Elige la clase positiva', options=classes)
    y = (y_raw == pos_label).astype(int)

    X = df[features]
    X = pd.get_dummies(X, drop_first=True)

    # Filtrar filas con NaN en la variable objetivo y asegurar que existan ambas clases
    mask = y_raw.notna()
    X = X[mask]
    y = y[mask]
    uniq = np.unique(y)
    if uniq.size < 2:
        st.error('La variable objetivo debe tener ambas clases presentes (0 y 1) para calcular matriz de confusión y ROC.')
        return

    # Comprobar que cada clase tenga al menos 2 muestras para poder estratificar en train/test
    counts = np.bincount(y.values, minlength=2)
    min_count = counts.min()

    if min_count < 2:
        st.warning('La clase menos poblada tiene solo 1 muestra; no es posible realizar train/test estratificado. Se entrenará y evaluará sobre TODO el conjunto. Para una evaluación más fiable, añade más datos de la clase minoritaria.')
        X_train, X_test, y_train, y_test = X.values, X.values, y.values, y.values
    else:
        X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=42, stratify=y.values)

    pipe = make_pipeline(StandardScaler() if scale else PolynomialFeatures(degree=1, include_bias=False), LogisticRegression(max_iter=1000)) if scale else LogisticRegression(max_iter=1000)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    thr = st.slider('Umbral de clasificación', 0.0, 1.0, 0.5, 0.01)
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = float('nan')
    # Nota: ROC/AUC se calculará más abajo solo si y_test contiene ambas clases

    st.markdown('La regresión logística modela el log-odds (log(p/(1-p))) y usa la función sigmoide σ(z)=1/(1+e^{-z}).')
    images = []
    interpretations = []
    col1, col2 = st.columns(2)
    with col1:
        # Plot ROC solo si y_test contiene ambas clases
        if np.unique(y_test).size >= 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.legend()
            st.pyplot(fig)
            images.append({'titulo': 'Curva ROC', 'bytes': fig_to_bytes(fig)})
            interpretations.append(f"La curva ROC (AUC={roc_auc:.3f}) resume la capacidad de discriminación del modelo.")
        else:
            st.info('No se puede calcular ROC/AUC porque y_test contiene una sola clase.')
            roc_auc = float('nan')

    with col2:
        # Matriz de confusión con labels fijos para asegurar forma 2x2
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        st.write('Matriz de confusión (labels=[0,1]):')
        st.write(cm)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, xticklabels=[0,1], yticklabels=[0,1])
        ax_cm.set_title('Matriz de confusión')
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')
        st.pyplot(fig_cm)
        images.append({'titulo': 'Matriz de confusión', 'bytes': fig_to_bytes(fig_cm)})
        interpretations.append(f"Con umbral={thr:.2f}, la matriz de confusión muestra TP/TN/FP/FN.")
        st.text('Reporte de clasificación:')
        st.text(classification_report(y_test, y_pred, labels=[0,1], zero_division=0))
        # Métricas por clase y su interpretación
        prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=[0,1], zero_division=0)
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        interpretations.append(f"Exactitud={acc:.3f} indica el porcentaje de predicciones correctas.")
        interpretations.append(f"Clase positiva (1): precisión={prec[1]:.3f} (de las predicciones positivas, cuántas son correctas), recall={rec[1]:.3f} (de los positivos reales, cuántos detecta), F1={f1[1]:.3f}.")
        interpretations.append(f"Clase negativa (0): precisión={prec[0]:.3f}, recall={rec[0]:.3f}, F1={f1[0]:.3f}.")
        interpretations.append(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}: más FP reduce la precisión; más FN reduce el recall. Ajusta el umbral para equilibrar.")

    st.write(f'Exactitud: {acc:.3f} | AUC: {roc_auc if not np.isnan(roc_auc) else "N/A"}')
    # Interpretación adicional del AUC y del umbral
    if not np.isnan(roc_auc):
        interpretations.append(f"AUC={roc_auc:.3f}: valores cercanos a 1 indican mejor capacidad de discriminación; el umbral {thr:.2f} controla el equilibrio entre TPR (sensibilidad) y FPR (tasa de falsos positivos).")
    # Si hubo muy pocos ejemplos en alguna clase, advertir sobre evaluación
    try:
        counts  # variable definida arriba
        if counts.min() < 2:
            interpretations.append('Debido a que la clase minoritaria tiene solo 1 muestra, se entrenó y evaluó sobre el mismo conjunto; las métricas pueden ser optimistas. Añade más datos para una evaluación más fiable.')
    except Exception:
        pass

    # Mostrar interpretación en la interfaz
    st.subheader('Interpretación')
    st.markdown('\n'.join([f"- {t}" for t in interpretations]))

    try:
        X_cols = X.columns.tolist()
        coef_df, intercept = extract_coefficients(pipe, X_cols)
        if coef_df is not None:
            coef_df['odds_ratio'] = np.exp(coef_df['coef'])
            st.subheader('Coeficientes (log-odds) y Odds Ratio')
            st.dataframe(coef_df)
            if intercept is not None:
                st.write(f'Intercepto: {intercept:.4f}')
            interpretations.append('Los coeficientes en log-odds indican el efecto de cada variable en la probabilidad. Un odds ratio > 1 incrementa la probabilidad de la clase positiva.')
    except Exception:
        pass

    st.session_state.setdefault('reports', []).append({
        'seccion': 'Logística',
        'items': [{
            'Exactitud': acc,
            'AUC': roc_auc,
            'Umbral': float(thr),
            'Precision_pos': float(prec[1]) if 'prec' in locals() else None,
            'Recall_pos': float(rec[1]) if 'rec' in locals() else None,
            'F1_pos': float(f1[1]) if 'f1' in locals() else None,
            'Precision_neg': float(prec[0]) if 'prec' in locals() else None,
            'Recall_neg': float(rec[0]) if 'rec' in locals() else None,
            'F1_neg': float(f1[0]) if 'f1' in locals() else None,
            'TP': int(tp) if 'tp' in locals() else None,
            'TN': int(tn) if 'tn' in locals() else None,
            'FP': int(fp) if 'fp' in locals() else None,
            'FN': int(fn) if 'fn' in locals() else None,
        }],
        'imagenes': images,
        'interpretaciones': interpretations,
        'fecha': datetime.now().isoformat()
    })