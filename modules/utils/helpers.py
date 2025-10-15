from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd

# Utilidades para extraer coeficientes y nombres de características

def _get_feature_names_for_pipeline(model, original_features):
    try:
        if hasattr(model, 'named_steps') and 'polynomialfeatures' in model.named_steps:
            poly = model.named_steps['polynomialfeatures']
            return poly.get_feature_names_out(original_features)
    except Exception:
        pass
    return original_features


def extract_coefficients(model, feature_names):
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

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()