import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import textwrap
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

DB_PATH = 'app.db'

# ---------------------- TEMA Y CABECERAS UI (Globales) ----------------------

def setup_ui_theme(theme: str = "light"):
    is_dark = theme == "dark"
    primary = "#5B8DEF"
    accent = "#F59E0B"
    bg_light = "#F7F9FC"
    bg_dark = "#0B1220"
    text_light = "#111827"
    text_dark = "#E5E7EB"
    card_light = "#FFFFFF"
    card_dark = "#111827"
    border_light = "#eaecef"
    border_dark = "#334155"

    bg = bg_dark if is_dark else bg_light
    text = text_dark if is_dark else text_light
    card = card_dark if is_dark else card_light
    border = border_dark if is_dark else border_light

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif !important;
        }}
        :root {{ --primary:{primary}; --accent:{accent}; --bg:{bg}; --text:{text}; --card:{card}; --border:{border}; }}
        .stApp {{ background: var(--bg); color: var(--text); }}
        .hero {{padding:1rem 1.25rem; border-radius:14px; background: linear-gradient(90deg,#5B8DEF 0%, #7C3AED 100%); color:#fff; margin: 0 0 1rem 0; box-shadow: 0 6px 20px rgba(91,141,239,.25); text-align:center;}}
        .hero .title {{font-size:1.4rem; font-weight:800; letter-spacing:.2px;}}
        .hero .subtitle {{opacity:.95; font-size:.95rem; margin-top:.25rem;}}
        .card {{background:var(--card); padding:1rem 1.2rem; border-radius:14px; border:1px solid var(--border); box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom:1rem;}}
        .sidebar-card {{padding:.7rem .8rem; background:{'#0F172A' if is_dark else '#eef2ff'}; border-radius:12px; margin-bottom:.75rem; border:1px solid {('#1F2937' if is_dark else '#dbeafe')}; color: var(--text);}}
        .badge {{display:inline-block; background:{'#334155' if is_dark else '#e5e7eb'}; color:{'#E5E7EB' if is_dark else '#111827'}; padding:.18rem .5rem; border-radius:8px; font-size:.8rem; margin-right:.35rem;}}
        .section-title {{font-size:1.2rem; font-weight:700; margin-bottom:.4rem; color:var(--text);}}
        .stButton>button {{background:var(--primary); color:#fff; border:none; border-radius:10px; padding:.5rem 1rem; font-weight:600; box-shadow: 0 1px 3px rgba(0,0,0,.15);}}
        .stButton>button:hover {{background:#4A76D1}}
        .download-btn .stButton>button {{background:var(--accent);}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", emoji: str = ""):
    st.markdown(
        f"""
        <div class='hero'>
            <div class='title'>{emoji} {title}</div>
            <div class='subtitle'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------- UTILIDADES DB ----------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        '''CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               email TEXT UNIQUE NOT NULL,
               nombre TEXT NOT NULL,
               codigo TEXT NOT NULL,
               password_hash TEXT NOT NULL,
               created_at TEXT NOT NULL
           )'''
    )
    conn.commit()
    conn.close()


def get_db():
    return sqlite3.connect(DB_PATH)


def hash_password(password: str) -> str:
    return pbkdf2_sha256.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return pbkdf2_sha256.verify(password, hashed)
    except Exception:
        return False


def register_user(email, nombre, codigo, password) -> tuple[bool, str]:
    if not email or not nombre or not codigo or not password:
        return False, 'Todos los campos son obligatorios.'
    if not codigo.isdigit() or len(codigo) != 6:
        return False, 'El código de estudiante debe tener exactamente 6 dígitos.'
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users(email, nombre, codigo, password_hash, created_at) VALUES (?,?,?,?,?)',
                  (email.strip(), nombre.strip(), codigo.strip(), hash_password(password), datetime.utcnow().isoformat()))
        conn.commit()
        return True, 'Registro exitoso. Ya puedes iniciar sesión.'
    except sqlite3.IntegrityError:
        return False, 'El correo ya está registrado.'
    finally:
        conn.close()


def login_user(email, password):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, email, nombre, codigo, password_hash FROM users WHERE email=?', (email.strip(),))
    row = c.fetchone()
    conn.close()
    if row and verify_password(password, row[4]):
        return {'id': row[0], 'email': row[1], 'nombre': row[2], 'codigo': row[3]}
    return None

# ---------------------- UI AUTENTICACIÓN ----------------------

def show_auth():
    page_header('Sistema de Aprendizaje Supervisado', 'JUAN CARLOS ANQUISE VARGAS<br/>CODIGO: 191062<br/>Explora modelos supervisados paso a paso.', '🤖')
    tabs = st.tabs(['Iniciar sesión', 'Registrarse'])
    
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader('Iniciar sesión')
        email = st.text_input('Correo', key='login_email')
        password = st.text_input('Contraseña', type='password', key='login_password')
        if st.button('Ingresar'):
            user = login_user(email, password)
            if user:
                st.session_state['user'] = user
                st.success(f"Bienvenido, {user['nombre']}!")
                st.rerun()
            else:
                st.error('Credenciales inválidas.')
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader('Registrarse')
        nombre = st.text_input('Nombre completo', key='register_nombre')
        email_r = st.text_input('Correo institucional', key='register_email')
        codigo = st.text_input('Código de estudiante (6 dígitos)', key='register_codigo')
        password_r = st.text_input('Contraseña', type='password', key='register_password')
        if st.button('Crear cuenta'):
            ok, msg = register_user(email_r, nombre, codigo, password_r)
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- UI UTILIDADES DE DATOS ----------------------

def load_sample_dataset_local():
    try:
        df = sns.load_dataset('mpg')
        # Ajuste de nombres más claros
        df = df.rename(columns={'mpg': 'consumo', 'displacement': 'desplazamiento', 'acceleration': 'aceleracion'})
        return df
    except Exception:
        return pd.DataFrame()


def data_option_local():
    page_header('Opción 1: Carga y preparación de datos', 'Carga, limpieza y codificación de variables.', '📦')
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown('Sube un archivo CSV/Excel o usa el dataset de ejemplo (rendimiento de automóviles).')

    uploaded = st.file_uploader('Subir CSV o Excel', type=['csv', 'xlsx'])
    df = None
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f'Error al leer archivo: {e}')
    else:
        if st.button('Usar dataset de ejemplo (automóviles)'):
            df = load_sample_dataset()

    if df is None or df.empty:
        st.info('Aún no hay datos cargados.')
        return None

    st.subheader('Vista previa')
    st.dataframe(df.head())

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

    # Edición de categorías (renombrar valores) antes de codificar
    cat_cols_current = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols_current:
        st.subheader('Editar/renombrar categorías')
        col_sel = st.selectbox('Columna categórica a editar', options=cat_cols_current)
        if col_sel:
            if col_sel not in df.columns:
                st.warning('La columna seleccionada ya no existe en el DataFrame (posiblemente fue codificada). Selecciona otra.')
            else:
                vals = sorted(pd.Series(df[col_sel].dropna().unique()).astype(str).tolist())
                old_val = st.selectbox('Valor a renombrar', options=vals)
                new_val = st.text_input('Nuevo nombre para la categoría seleccionada', key='cat_edit_new_name')
                if st.button('Renombrar categoría') and new_val:
                    df[col_sel] = df[col_sel].astype(str).replace({old_val: new_val})
                    st.success(f'Se renombró "{old_val}" a "{new_val}" en la columna {col_sel}.')
                    st.dataframe(df[[col_sel]].head())

        # Codificación binaria personalizada (ej. X11: A=0, M=1)
        st.subheader('Codificación binaria personalizada')
        col_bin = st.selectbox('Columna binaria a codificar', options=cat_cols_current, key='bin_col_sel')
        if col_bin:
            if col_bin not in df.columns:
                st.warning('La columna seleccionada ya no existe (posible codificación previa).')
            else:
                uniques = sorted(pd.Series(df[col_bin].dropna().unique()).astype(str).tolist())
                st.write(f'Valores detectados en {col_bin}: {uniques}')
                # Preselección si existen A y M
                default_zero = 'A' if 'A' in uniques else (uniques[0] if uniques else '')
                default_one = 'M' if 'M' in uniques else (uniques[1] if len(uniques) > 1 else '')
                val_zero = st.selectbox('Valor que será 0', options=uniques, index=(uniques.index(default_zero) if default_zero in uniques else 0), key='map_zero')
                val_one = st.selectbox('Valor que será 1', options=uniques, index=(uniques.index(default_one) if default_one in uniques else (1 if len(uniques) > 1 else 0)), key='map_one')
                if val_zero == val_one:
                    st.warning('Selecciona valores distintos para 0 y 1.')
                else:
                    if st.button('Aplicar mapeo 0/1'):
                        df[col_bin] = df[col_bin].astype(str).replace({val_zero: 0, val_one: 1})
                        df[col_bin] = pd.to_numeric(df[col_bin], errors='coerce').astype('Int64')
                        st.success(f'Columna {col_bin} codificada: {val_zero}→0, {val_one}→1.')
                        st.dataframe(df[[col_bin]].head())
    else:
        st.info('No se detectaron columnas categóricas para editar.')

    st.subheader('Codificación de variables categóricas')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        drop_first = st.checkbox('Eliminar una categoría por variable para evitar colinealidad (drop_first)', value=True)
        if st.button('Convertir categóricas a numéricas (One-Hot)'):
            df = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
            st.success('Conversión realizada.')
            st.dataframe(df.head())
            # Permitir eliminar columnas dummy específicas para evitar colinealidad
            posibles_dummies = [c for c in df.columns if any(c.startswith(col + '_') for col in cat_cols)]
            if posibles_dummies:
                to_drop = st.multiselect('Selecciona columnas dummy a eliminar', options=posibles_dummies)
                if st.button('Eliminar columnas seleccionadas') and to_drop:
                    df = df.drop(columns=to_drop, errors='ignore')
                    st.success('Columnas dummy eliminadas.')
                    st.dataframe(df.head())
    else:
        st.info('No se detectaron columnas categóricas.')

    st.session_state['df'] = df
    return df

# ---------------------- REGRESIÓN ----------------------

def regression_option_local():
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    models = []
    images = []
    interpretations = []
    # Regresión múltiple lineal
    lr = make_pipeline(StandardScaler() if scale else PolynomialFeatures(degree=1, include_bias=False), LinearRegression()) if scale else LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    models.append(('Regresión lineal', lr, y_pred_lr))

    # Transformación polinómica
    deg = st.slider('Grado polinómico', 2, 5, 2)
    poly_model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), LinearRegression())
    poly_model.fit(X_train, y_train)
    y_pred_poly = poly_model.predict(X_test)
    models.append((f'Regresión polinómica (grado {deg})', poly_model, y_pred_poly))

    # Kernel RBF y polinómico via SVR
    C = st.number_input('C (SVR)', 0.1, 100.0, 1.0)
    gamma = st.selectbox('gamma (SVR-RBF)', options=['scale', 'auto'])
    svr_rbf = make_pipeline(StandardScaler() if scale else StandardScaler(), SVR(kernel='rbf', C=C, gamma=gamma))
    svr_rbf.fit(X_train, y_train)
    y_pred_rbf = svr_rbf.predict(X_test)
    models.append(('SVR (RBF)', svr_rbf, y_pred_rbf))

    degree_svr = st.slider('Grado kernel polinómico (SVR)', 2, 5, 3)
    svr_poly = make_pipeline(StandardScaler() if scale else StandardScaler(), SVR(kernel='poly', C=C, degree=degree_svr))
    svr_poly.fit(X_train, y_train)
    y_pred_svr_poly = svr_poly.predict(X_test)
    models.append((f'SVR (polinómico grado {degree_svr})', svr_poly, y_pred_svr_poly))

    # Regularización: Ridge y Lasso
    alpha = st.number_input('alpha (Ridge/Lasso)', 0.0001, 10.0, 1.0)
    ridge = make_pipeline(PolynomialFeatures(degree=1, include_bias=False), Ridge(alpha=alpha))
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    models.append(('Ridge', ridge, y_pred_ridge))

    lasso = make_pipeline(PolynomialFeatures(degree=1, include_bias=False), Lasso(alpha=alpha, max_iter=10000))
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

# ---------------------- CLASIFICACIÓN LOGÍSTICA ----------------------

def logistic_option_local():
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

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=42)

    pipe = make_pipeline(StandardScaler() if scale else PolynomialFeatures(degree=1, include_bias=False), LogisticRegression(max_iter=1000)) if scale else LogisticRegression(max_iter=1000)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    thr = st.slider('Umbral de clasificación', 0.0, 1.0, 0.5, 0.01)
    y_pred = (y_prob >= thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    st.write(f'Exactitud: {acc:.3f} | AUC: {roc_auc:.3f}')
    st.markdown('La regresión logística modela el log-odds (log(p/(1-p))) y usa la función sigmoide σ(z)=1/(1+e^{-z}).')
    images = []
    interpretations = []
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend()
        st.pyplot(fig)
        images.append({'titulo': 'Curva ROC', 'bytes': fig_to_bytes(fig)})
        interpretations.append(f"La curva ROC (AUC={roc_auc:.3f}) resume la capacidad de discriminación del modelo. AUC cercano a 1 indica excelente separación entre clases.")
    with col2:
        cm = confusion_matrix(y_test, y_pred)
        st.write('Matriz de confusión:')
        st.write(cm)
        # Visualización tipo heatmap para PDF
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_title('Matriz de confusión')
        ax_cm.set_xlabel('Predicción')
        ax_cm.set_ylabel('Real')
        st.pyplot(fig_cm)
        images.append({'titulo': 'Matriz de confusión', 'bytes': fig_to_bytes(fig_cm)})
        interpretations.append(f"Con umbral={thr:.2f}, la matriz de confusión muestra verdaderos positivos y negativos, así como falsos positivos y negativos. Una alta diagonal (valores grandes en [0,0] y [1,1]) implica buen desempeño.")
        st.text('Reporte de clasificación:')
        st.text(classification_report(y_test, y_pred))

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
        'items': [{'Exactitud': acc, 'AUC': roc_auc, 'Umbral': float(thr)}],
        'imagenes': images,
        'interpretaciones': interpretations,
        'fecha': datetime.now().isoformat()
    })

# ---------------------- ÁRBOLES DE DECISIÓN ----------------------

def tree_option_local():
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
        # Matplotlib bar para PDF
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
    # Matplotlib bar para PDF
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

def report_option_local():
    st.header('Opción 5: Reporte y Descarga en PDF')
    reports = st.session_state.get('reports', [])
    if not reports:
        st.info('Aún no hay resultados para reportar. Ejecuta algunos modelos primero.')
        return

    st.write('Vista previa del contenido del reporte:')
    for r in reports:
        st.write(f"- Sección: {r['seccion']} ({r['fecha']})")
        for it in r.get('items', []):
            st.write(f"  * {it}")
        # Mostrar previsualización de imágenes
        for im in r.get('imagenes', []):
            st.image(im['bytes'], caption=im.get('titulo', 'Figura'), use_column_width=True)
        # Mostrar interpretaciones
        interps = r.get('interpretaciones', [])
        if interps:
            st.write('Interpretaciones:')
            for txt in interps:
                st.markdown(f"- {txt}")

    # Determinar mejores modelos
    best_reg = None
    best_log = None
    best_tree_cls = None
    best_tree_reg = None
    for r in reports:
        sec = r.get('seccion')
        items = r.get('items', [])
        if sec == 'Regresión':
            for it in items:
                if isinstance(it, dict) and 'R2' in it:
                    if best_reg is None or it['R2'] > best_reg['R2']:
                        best_reg = it
        elif sec == 'Logística':
            for it in items:
                if isinstance(it, dict) and 'AUC' in it:
                    if best_log is None or it['AUC'] > best_log['AUC']:
                        best_log = it
        elif sec == 'Árbol (Clasificación)':
            for it in items:
                if isinstance(it, dict) and 'Acc_test' in it:
                    if best_tree_cls is None or it['Acc_test'] > best_tree_cls['Acc_test']:
                        best_tree_cls = it
        elif sec == 'Árbol (Regresión)':
            for it in items:
                if isinstance(it, dict) and 'R2_test' in it:
                    if best_tree_reg is None or it['R2_test'] > best_tree_reg['R2_test']:
                        best_tree_reg = it

    st.subheader('Mejores modelos por caso')
    if best_reg is not None:
        st.write(f"Regresión: {best_reg.get('modelo', 'N/A')} (R²={best_reg.get('R2', 0):.3f}, RMSE={best_reg.get('RMSE', 0):.3f})")
    if best_log is not None:
        st.write(f"Logística: AUC={best_log.get('AUC', 0):.3f}, Exactitud={best_log.get('Exactitud', 0):.3f}")
    if best_tree_cls is not None:
        st.write(f"Árbol (Clasificación): Acc_test={best_tree_cls.get('Acc_test', 0):.3f}, max_depth={best_tree_cls.get('max_depth', 'N/A')}")
    if best_tree_reg is not None:
        st.write(f"Árbol (Regresión): R2_test={best_tree_reg.get('R2_test', 0):.3f}, max_depth={best_tree_reg.get('max_depth', 'N/A')}")

    if st.button('Generar y descargar PDF'):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 2*cm
        left = 2*cm
        right_margin = 2*cm
        usable_w = width - left - right_margin
        c.setFont('Helvetica-Bold', 14)
        c.drawString(left, y, 'Reporte de Modelos - Aprendizaje Supervisado')
        y -= 1*cm

        # Resumen: mejores modelos por caso
        c.setFont('Helvetica-Bold', 12)
        c.drawString(left, y, 'Mejores modelos por caso:')
        y -= 0.7*cm
        c.setFont('Helvetica', 10)
        if best_reg is not None:
            c.drawString(left, y, f"Regresión: {best_reg.get('modelo', 'N/A')} (R²={best_reg.get('R2', 0):.3f}, RMSE={best_reg.get('RMSE', 0):.3f})")
            y -= 0.5*cm
        if best_log is not None:
            c.drawString(left, y, f"Logística: AUC={best_log.get('AUC', 0):.3f}, Exactitud={best_log.get('Exactitud', 0):.3f}")
            y -= 0.5*cm
        if best_tree_cls is not None:
            c.drawString(left, y, f"Árbol (Clasificación): Acc_test={best_tree_cls.get('Acc_test', 0):.3f}, max_depth={best_tree_cls.get('max_depth', 'N/A')}")
            y -= 0.5*cm
        if best_tree_reg is not None:
            c.drawString(left, y, f"Árbol (Regresión): R2_test={best_tree_reg.get('R2_test', 0):.3f}, max_depth={best_tree_reg.get('max_depth', 'N/A')}")
            y -= 0.5*cm
        if y < 2*cm:
            c.showPage(); y = height - 2*cm

        # Detalle con imágenes e interpretaciones
        c.setFont('Helvetica-Bold', 12)
        c.drawString(left, y, 'Detalle de resultados e imágenes:')
        y -= 0.6*cm
        c.setFont('Helvetica', 10)
        for r in reports:
            c.drawString(left, y, f"Sección: {r['seccion']} ({r['fecha']})")
            y -= 0.6*cm
            # Items
            for it in r.get('items', []):
                c.drawString(left + 0.5*cm, y, str(it))
                y -= 0.5*cm
                if y < 4*cm:
                    c.showPage(); y = height - 2*cm
            # Imágenes
            for im in r.get('imagenes', []):
                try:
                    img_reader = ImageReader(BytesIO(im['bytes']))
                    iw, ih = img_reader.getSize()
                    scale = usable_w / iw
                    draw_h = ih * scale
                    if y - draw_h < 2*cm:
                        c.showPage(); y = height - 2*cm
                    c.drawImage(img_reader, left, y - draw_h, width=usable_w, height=draw_h)
                    y -= draw_h + 0.4*cm
                    c.drawString(left, y, f"Figura: {im.get('titulo','Imagen')}")
                    y -= 0.5*cm
                except Exception:
                    # Si falla la imagen, continuar
                    pass
            # Interpretaciones
            interps = r.get('interpretaciones', [])
            if interps:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(left, y, 'Interpretaciones:')
                y -= 0.4*cm
                c.setFont('Helvetica', 10)
                for txt in interps:
                    # envolver texto
                    for line in textwrap.wrap(txt, width=100):
                        c.drawString(left + 0.5*cm, y, f"- {line}")
                        y -= 0.4*cm
                        if y < 2*cm:
                            c.showPage(); y = height - 2*cm
            if y < 2*cm:
                c.showPage(); y = height - 2*cm
        c.save()
        buffer.seek(0)
        st.download_button('Descargar PDF', data=buffer, file_name='reporte_modelos.pdf', mime='application/pdf')

from modules.ui.theme import setup_ui_theme, page_header
from modules.options.data import data_option
from modules.options.regression import regression_option
from modules.options.logistic import logistic_option
from modules.options.decision_tree import tree_option
from modules.options.report import report_option
from modules.utils.helpers import _get_feature_names_for_pipeline, extract_coefficients, fig_to_bytes

def main():
    init_db()
    st.set_page_config(page_title='Aprendizaje Supervisado', layout='wide', page_icon='🧠')
    # Apply persisted theme preference
    dark_mode_pref = st.session_state.get('dark_mode', False)
    setup_ui_theme('dark' if dark_mode_pref else 'light')
    
    if 'user' not in st.session_state:
        show_auth()
        return

    user = st.session_state['user']
    with st.sidebar:
        # Theme toggle
        dark_mode_cb = st.checkbox('Modo oscuro 🌙', value=dark_mode_pref)
        if dark_mode_cb != dark_mode_pref:
            st.session_state['dark_mode'] = dark_mode_cb
            setup_ui_theme('dark' if dark_mode_cb else 'light')
        
        st.markdown(
            f"""
            <div class='sidebar-card'>
                <div style='font-weight:700; font-size:1rem;'>👤 {user['nombre']}</div>
                <div class='badge'>Código: {user['codigo']}</div>
                <div class='badge'>📧 {user['email']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        choice = st.radio('Opciones del Dashboard', options=['Opción 1: Datos 📦', 'Opción 2: Regresión 📈', 'Opción 3: Logística 🧭', 'Opción 4: Árbol de decisión 🌳', 'Opción 5: Reporte 📝'])
        if st.button('Cerrar sesión'):
            st.session_state.pop('user', None)
            st.rerun()

    if choice.startswith('Opción 1'):
        data_option()
    elif choice.startswith('Opción 2'):
        regression_option()
    elif choice.startswith('Opción 3'):
        logistic_option()
    elif choice.startswith('Opción 4'):
        tree_option()
    else:
        report_option()




# Utilidades para extraer coeficientes y nombres de características

def _get_feature_names_for_pipeline_local(model, original_features):
    try:
        if hasattr(model, 'named_steps') and 'polynomialfeatures' in model.named_steps:
            poly = model.named_steps['polynomialfeatures']
            return poly.get_feature_names_out(original_features)
    except Exception:
        pass
    return original_features


def extract_coefficients_local(model, feature_names):
    estimator = model
    if hasattr(model, 'named_steps'):
        for name in reversed(list(model.named_steps.keys())):
            step = model.named_steps[name]
            if hasattr(step, 'coef_'):
                estimator = step
                break
        feature_names = _get_feature_names_for_pipeline(model, feature_names)
    if hasattr(estimator, 'coef_'):
        coef = estimator.coef_
        if hasattr(coef, 'ndim') and coef.ndim > 1:
            coef = coef.ravel()
        intercept = getattr(estimator, 'intercept_', None)
        df_coef = pd.DataFrame({'feature': feature_names[:len(coef)], 'coef': coef})
        return df_coef, intercept
    return None, None


# Utilidad: convertir figura matplotlib a bytes (PNG)
def fig_to_bytes_local(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _get_feature_names_for_pipeline_local_dup(model, original_features):
    try:
        if hasattr(model, 'named_steps') and 'polynomialfeatures' in model.named_steps:
            poly = model.named_steps['polynomialfeatures']
            return poly.get_feature_names_out(original_features)
    except Exception:
        pass
    return original_features


def extract_coefficients_local_dup(model, feature_names):
    estimator = model
    if hasattr(model, 'named_steps'):
        for name in reversed(list(model.named_steps.keys())):
            step = model.named_steps[name]
            if hasattr(step, 'coef_'):
                estimator = step
                break
        feature_names = _get_feature_names_for_pipeline(model, feature_names)
    if hasattr(estimator, 'coef_'):
        coef = estimator.coef_
        if hasattr(coef, 'ndim') and coef.ndim > 1:
            coef = coef.ravel()
        intercept = getattr(estimator, 'intercept_', None)
        df_coef = pd.DataFrame({'feature': feature_names[:len(coef)], 'coef': coef})
        return df_coef, intercept
    return None, None


# Utilidad: convertir figura matplotlib a bytes (PNG)
def fig_to_bytes_local_dup(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    main()