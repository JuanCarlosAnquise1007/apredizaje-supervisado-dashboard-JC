import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

from modules.utils.helpers import fig_to_bytes, extract_coefficients


def regression_option():
    st.header('Opción 2: Modelos de Regresión')
    df = st.session_state.get('df')
    if df is None or df.empty:
        st.warning('Primero carga/prepara datos en Opción 1.')
        return

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox('Variable objetivo (numérica)', options=numeric_cols)
    features = st.multiselect('Características (X)', options=[c for c in numeric_cols if c != target])
    test_size = st.slider('Proporción de prueba', 0.1, 0.5, 0.2, 0.05)
    scale = st.checkbox('Estandarizar características')

    if not target or not features:
        st.info('Selecciona objetivo y características.')
        return

    X = df[features].values
    y = df[target].values

    # Eliminar filas con NaN en y (SimpleImputer no aplica a y)
    mask = ~np.isnan(y)
    if mask.sum() < len(y):
        st.info(f'Se eliminaron {len(y) - mask.sum()} filas con objetivo NaN.')
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    models = []
    images = []
    interpretations = []
    # Regresión múltiple lineal con imputación de NaN
    lr_steps = [SimpleImputer(strategy='median')]
    if scale:
        lr_steps.append(StandardScaler())
    lr_steps.append(LinearRegression())
    lr = make_pipeline(*lr_steps)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    models.append(('Regresión lineal', lr, y_pred_lr))

    # Transformación polinómica con imputación
    deg = st.slider('Grado polinómico', 2, 5, 2)
    poly_steps = [SimpleImputer(strategy='median'), PolynomialFeatures(degree=deg, include_bias=False)]
    if scale:
        poly_steps.append(StandardScaler())
    poly_steps.append(LinearRegression())
    poly_model = make_pipeline(*poly_steps)
    poly_model.fit(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)
    models.append((f'Regresión polinómica (grado {deg})', poly_model, y_pred_poly))

    # Kernel RBF y polinómico via SVR con imputación
    C = st.number_input('C (SVR)', 0.1, 100.0, 1.0)
    gamma = st.selectbox('gamma (SVR-RBF)', options=['scale', 'auto'])
    svr_rbf_steps = [SimpleImputer(strategy='median')]
    if scale:
        svr_rbf_steps.append(StandardScaler())
    svr_rbf_steps.append(SVR(kernel='rbf', C=C, gamma=gamma))
    svr_rbf = make_pipeline(*svr_rbf_steps)
    svr_rbf.fit(X_train, y_train)
    y_pred_rbf = svr_rbf.predict(X_test)
    models.append(('SVR (RBF)', svr_rbf, y_pred_rbf))

    degree_svr = st.slider('Grado kernel polinómico (SVR)', 2, 5, 3)
    svr_poly_steps = [SimpleImputer(strategy='median')]
    if scale:
        svr_poly_steps.append(StandardScaler())
    svr_poly_steps.append(SVR(kernel='poly', C=C, degree=degree_svr))
    svr_poly = make_pipeline(*svr_poly_steps)
    svr_poly.fit(X_train, y_train)
    y_pred_svr_poly = svr_poly.predict(X_test)
    models.append((f'SVR (polinómico grado {degree_svr})', svr_poly, y_pred_svr_poly))

    # Regularización: Ridge y Lasso con imputación
    alpha = st.number_input('alpha (Ridge/Lasso)', 0.0001, 10.0, 1.0)
    ridge_steps = [SimpleImputer(strategy='median'), PolynomialFeatures(degree=1, include_bias=False)]
    if scale:
        ridge_steps.append(StandardScaler())
    ridge_steps.append(Ridge(alpha=alpha))
    ridge = make_pipeline(*ridge_steps)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    models.append(('Ridge', ridge, y_pred_ridge))

    lasso_steps = [SimpleImputer(strategy='median'), PolynomialFeatures(degree=1, include_bias=False)]
    if scale:
        lasso_steps.append(StandardScaler())
    lasso_steps.append(Lasso(alpha=alpha, max_iter=10000))
    lasso = make_pipeline(*lasso_steps)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    models.append(('Lasso', lasso, y_pred_lasso))

    st.subheader('Resultados')
    best_name, best_r2 = None, -np.inf
    report_items = []
    for name, mdl, y_pred in models:
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'{name}: R²={r2:.3f}, RMSE={rmse:.3f}')
            try:
                coef_df, intercept = extract_coefficients(mdl, features)
                if coef_df is not None:
                    st.write('Coeficientes:')
                    st.dataframe(coef_df)
                    if intercept is not None:
                        st.write(f'Intercepto: {intercept:.4f}')
            except Exception:
                pass
        with col2:
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel('Real')
            ax.set_ylabel('Predicho')
            ax.set_title(name)
            st.pyplot(fig)
            # Capturar imagen para PDF
            img_bytes = fig_to_bytes(fig)
            images.append({'titulo': f'Dispersión {name}', 'bytes': img_bytes})
            interpretations.append(f"{name}: R²={r2:.3f}, RMSE={rmse:.3f}. Puntos cercanos a la diagonal indican buen ajuste. Un R² más alto y RMSE más bajo sugieren mejor desempeño.")
        if r2 > best_r2:
            best_r2, best_name = r2, name
        report_items.append({'modelo': name, 'R2': r2, 'RMSE': rmse})

    st.success(f'Mejor modelo según R²: {best_name} (R²={best_r2:.3f})')
    st.session_state.setdefault('reports', []).append({
        'seccion': 'Regresión',
        'items': report_items,
        'imagenes': images,
        'interpretaciones': interpretations,
        'fecha': datetime.now().isoformat()
    })