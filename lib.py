import pandas as pd
import numpy as np

def extract_gpu_info(gpu_string):
    if pd.isna(gpu_string) or gpu_string is None or gpu_string == "":
        return None, None
    parts = gpu_string.split()
    if len(parts) == 0:
        return None, None
    elif len(parts) == 1:
        return parts[0], None
    else:
        return parts[0], ' '.join(parts[1:])

def safe_transform(le, value):
    if value is None or value == 'Unknown':
        return -1
    try:
        return le.transform([value])[0]
    except ValueError:
        return -1

def predict_compatibility(title, gpu, distribution, cpu, ram, kernel, model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler):
    gpu_manufacturer, gpu_model = extract_gpu_info(gpu)

    test_data = pd.DataFrame({
        'title': [title if title else 'Unknown'],
        'gpu_manufacturer': [gpu_manufacturer if gpu_manufacturer else 'Unknown'],
        'gpu_model': [gpu_model if gpu_model else 'Unknown'],
        'distribution': [distribution if distribution else 'Unknown'],
        'cpu': [cpu if cpu else 'Unknown'],
        'ram': [ram if ram else 'Unknown'],
        'kernel': [kernel if kernel else 'Unknown']
    })

    test_data['title_encoded'] = safe_transform(le_title, title)
    test_data['gpu_manufacturer_encoded'] = safe_transform(le_gpu_manufacturer, gpu_manufacturer)
    test_data['gpu_model_encoded'] = safe_transform(le_gpu_model, gpu_model)
    test_data['distribution_encoded'] = safe_transform(le_distribution, distribution)
    test_data['cpu_encoded'] = safe_transform(le_cpu, cpu)
    test_data['ram_encoded'] = safe_transform(le_ram, ram)
    test_data['kernel_encoded'] = safe_transform(le_kernel, kernel)

    unknown_labels = []
    partial_info = []
    for col, orig_col in zip(['title_encoded', 'gpu_manufacturer_encoded', 'gpu_model_encoded', 'distribution_encoded', 'cpu_encoded', 'ram_encoded', 'kernel_encoded'],
                             ['title', 'gpu_manufacturer', 'gpu_model', 'distribution', 'cpu', 'ram', 'kernel']):
        if test_data[col].iloc[0] == -1:
            if orig_col == 'gpu_model' and gpu_manufacturer is not None:
                partial_info.append("GPU (nur Hersteller)")
            else:
                unknown_labels.append(f"{orig_col.capitalize()}: Unbekannt")

    X_pred = np.array([
        test_data['title_encoded'] * 3,
        test_data['gpu_manufacturer_encoded'] * 2,
        test_data['gpu_model_encoded'],
        test_data['distribution_encoded'],
        test_data['cpu_encoded'],
        test_data['ram_encoded'],
        test_data['kernel_encoded']
    ]).T

    X_pred_scaled = scaler.transform(X_pred)
    predicted_score = max(0, model.predict(X_pred_scaled)[0][0])

    if unknown_labels or partial_info:
        completeness_factor = 1 - (len(unknown_labels) + 0.5 * len(partial_info)) / 7
        adjusted_score = max(0, predicted_score * completeness_factor)
    else:
        adjusted_score = predicted_score

    if adjusted_score < 0.1:
        compatibility_level = "Sehr schlecht (läuft wahrscheinlich nicht)"
    elif adjusted_score < 0.3:
        compatibility_level = "Schlecht"
    elif adjusted_score < 0.5:
        compatibility_level = "Mäßig"
    elif adjusted_score < 0.7:
        compatibility_level = "Gut"
    else:
        compatibility_level = "Hervorragend"

    gpu_info = f"{gpu_manufacturer} {gpu_model}".strip() if gpu_model else gpu_manufacturer
    specs = [f"{gpu_info} GPU" if gpu else None,
             f"auf {distribution}" if distribution else None,
             f"mit {cpu}" if cpu else None,
             f"{ram} RAM" if ram else None,
             f"{kernel} Kernel" if kernel else None]
    specs = [spec for spec in specs if spec]

    result = {
        "title": title,
        "compatibility_level": compatibility_level,
        "quality_score": adjusted_score,
        "specs": specs,
        "unknown_labels": unknown_labels,
        "partial_info": partial_info
    }

    return result
