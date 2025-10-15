import streamlit as st
from io import BytesIO
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from modules.ui.theme import page_header
import pandas as pd


def report_option():
    page_header('Opción 5: Reporte y Descarga en PDF', 'Resumen, interpretación y descarga paso a paso.', '📄')
    reports = st.session_state.get('reports', [])
    if not reports:
        st.info('Aún no hay resultados para reportar. Ejecuta algunos modelos primero.')
        return

    # Estilos para centrar y mejorar presentación
    st.markdown(
        """
        <style>
        .centered { max-width: 1000px; margin: 0 auto; }
        .report-section { margin-bottom: 1rem; }
        .report-title { text-align:center; font-weight:700; font-size:1.05rem; margin-bottom: .5rem; }
        .kvs { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:.4rem .8rem; }
        .kv { background: var(--card); border:1px solid var(--border); border-radius:10px; padding:.5rem .7rem; }
        .kv .k { font-weight:600; }
        .step-title { font-weight:700; margin:.4rem 0 .3rem 0; }
        .center-text { text-align:center; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='card centered'>", unsafe_allow_html=True)
    st.subheader('Resumen y vista previa')
    # Vista previa organizada por sección, centrado y paso a paso
    for r in reports:
        with st.expander(f"Sección: {r['seccion']} ({r['fecha']})", expanded=False):
            st.markdown("<div class='report-section'>", unsafe_allow_html=True)
            # Paso 1: Métricas clave / Items
            items = r.get('items', [])
            if items:
                st.markdown("<div class='step-title'>Paso 1: Métricas clave</div>", unsafe_allow_html=True)
                # Convertir items en tabla presentable
                def _items_to_df(items):
                    rows = []
                    for it in items:
                        if isinstance(it, dict):
                            # Formateo de valores numéricos a 3 decimales
                            formatted = {}
                            for k, v in it.items():
                                if isinstance(v, (int, float)) and v is not None:
                                    formatted[k] = float(f"{v:.3f}")
                                else:
                                    formatted[k] = v
                            rows.append(formatted)
                        else:
                            rows.append({'Detalle': str(it)})
                    return pd.DataFrame(rows)
                df_items = _items_to_df(items)
                st.dataframe(df_items, use_container_width=True)

            # Paso 2: Visualizaciones
            images = r.get('imagenes', [])
            if images:
                st.markdown("<div class='step-title'>Paso 2: Visualizaciones</div>", unsafe_allow_html=True)
                cols = st.columns(2)
                for idx, im in enumerate(images):
                    with cols[idx % 2]:
                        st.image(im['bytes'], caption=im.get('titulo', 'Figura'), use_column_width=True)

            # Paso 3: Interpretaciones
            interps = r.get('interpretaciones', [])
            if interps:
                st.markdown("<div class='step-title'>Paso 3: Interpretaciones</div>", unsafe_allow_html=True)
                st.markdown('\n'.join([f"- {txt}" for txt in interps]))
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Cálculo de mejores modelos
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

    # Tarjeta de mejores modelos, centrada
    st.markdown("<div class='card centered'>", unsafe_allow_html=True)
    st.subheader('Mejores modelos por caso')
    if best_reg is not None:
        st.markdown(f"<div class='center-text'>Regresión: {best_reg.get('modelo', 'N/A')} (R²={best_reg.get('R2', 0):.3f}, RMSE={best_reg.get('RMSE', 0):.3f})</div>", unsafe_allow_html=True)
    if best_log is not None:
        st.markdown(f"<div class='center-text'>Logística: AUC={best_log.get('AUC', 0):.3f}, Exactitud={best_log.get('Exactitud', 0):.3f}</div>", unsafe_allow_html=True)
    if best_tree_cls is not None:
        st.markdown(f"<div class='center-text'>Árbol (Clasificación): Acc_test={best_tree_cls.get('Acc_test', 0):.3f}, max_depth={best_tree_cls.get('max_depth', 'N/A')}</div>", unsafe_allow_html=True)
    if best_tree_reg is not None:
        st.markdown(f"<div class='center-text'>Árbol (Regresión): R2_test={best_tree_reg.get('R2_test', 0):.3f}, max_depth={best_tree_reg.get('max_depth', 'N/A')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Generación de PDF con mejor formato de items
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

        c.setFont('Helvetica-Bold', 12)
        c.drawString(left, y, 'Detalle de resultados e imágenes:')
        y -= 0.6*cm
        c.setFont('Helvetica', 10)
        for r in reports:
            c.drawString(left, y, f"Sección: {r['seccion']} ({r['fecha']})")
            y -= 0.6*cm
            for it in r.get('items', []):
                if isinstance(it, dict):
                    for k, v in it.items():
                        if isinstance(v, (int, float)) and v is not None:
                            v_str = f"{v:.3f}"
                        else:
                            v_str = str(v)
                        c.drawString(left + 0.5*cm, y, f"- {k}: {v_str}")
                        y -= 0.4*cm
                        if y < 4*cm:
                            c.showPage(); y = height - 2*cm
                else:
                    c.drawString(left + 0.5*cm, y, str(it))
                    y -= 0.5*cm
                    if y < 4*cm:
                        c.showPage(); y = height - 2*cm
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
                    pass
            interps = r.get('interpretaciones', [])
            if interps:
                c.setFont('Helvetica-Bold', 10)
                c.drawString(left, y, 'Interpretaciones:')
                y -= 0.4*cm
                c.setFont('Helvetica', 10)
                for txt in interps:
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