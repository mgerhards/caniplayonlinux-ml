import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import logging
import tarfile
import tempfile
import glob
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def calculate_quality_score(data):
    #if not isinstance(data, dict) or 'responses' not in data:
    #    print(f"Ungültige Eingabe für data: {data}")
    #    return 0  # Rückgabe von 0 für ungültige Eingaben

    responses = data['responses']
    score = 0
    max_score = 0

    quality_fields = [
        'audioFaults', 'graphicalFaults', 'inputFaults', 'performanceFaults',
        'saveGameFaults', 'significantBugs', 'stabilityFaults', 'windowingFaults'
    ]
    functionality_fields = ['installs', 'opens', 'startsPlay']

    logging.debug("Einzelne Bewertungen:")
    for field in quality_fields:
        if field in responses:
            max_score += 1
            if responses[field] == 'no':
                score += 1
            logging.debug(f"{field}: {responses[field]} (+{1 if responses[field] == 'no' else 0})")
        else:
            logging.debug(f"{field}: nicht gefunden")

    for field in functionality_fields:
        if field in responses:
            max_score += 4
            if responses[field] == 'no':
                return 0
            elif responses[field] == 'yes':
                score += 4
            logging.debug(f"{field}: {responses[field]} (+{4 if responses[field] == 'yes' else 0})")
        else:
            logging.debug(f"{field}: nicht gefunden")

    if 'triedOob' in responses:
        max_score += 5
        if responses['triedOob'] == 'yes':
            score += 5
        logging.debug(f"triedOob: {responses['triedOob']} (+{5 if responses['triedOob'] == 'yes' else 0})")
    else:
        logging.debug("triedOob: nicht gefunden")

    logging.debug(f"Zwischenstand - Score: {score}, Max Score: {max_score}")

    if max_score == 0:
        logging.debug("Warnung: Keine gültigen Felder gefunden.")
        return 0

    final_score = score / max_score
    logging.debug(f"Finaler Score: {final_score}")
    return final_score


def prepare_data(json_data):
    try:
        data = json_data  # Jetzt direkt die JSON-Daten verwenden
    except json.JSONDecodeError:
        logging.error(f"Fehler: Die JSON-Daten sind ungültig.")
        return None

    df = pd.DataFrame(data)

    # Extrahieren der relevanten Felder
    df['title'] = df['app'].apply(lambda x: x['title'])
    df['gpu'] = df['systemInfo'].apply(lambda x: x['gpu'])
    df['distribution'] = df['systemInfo'].apply(lambda x: x['os'])
    df['cpu'] = df['systemInfo'].apply(lambda x: x['cpu'])
    df['ram'] = df['systemInfo'].apply(lambda x: x['ram'])
    df['kernel'] = df['systemInfo'].apply(lambda x: x['kernel'])

    # Ausgabe der Rohdaten für "Destiny 2"
    debug_game_data = df[df['title'] == 'Destiny 2']
    total_quality_score = 0
    data_count = 0

    if not debug_game_data.empty:
        logging.info("Rohdaten für Destiny 2:")
        for index, row in debug_game_data.iterrows():
            quality_score = calculate_quality_score(row)
            total_quality_score += quality_score
            data_count += 1

            logging.info(f"Title: {row['title']}")
            logging.info(f"GPU: {row['gpu']}")
            logging.info(f"Distribution: {row['distribution']}")
            logging.info(f"CPU: {row['cpu']}")
            logging.info(f"RAM: {row['ram']}")
            logging.info(f"Kernel: {row['kernel']}")
            logging.info(f"Quality Score: {quality_score}")
            logging.info("Responses:")
            for key, value in row['responses'].items():
                logging.info(f"  {key}: {value}")
            logging.info("---")

        if data_count > 0:
            average_quality_score = total_quality_score / data_count
            logging.info(f"Durchschnittlicher Quality Score für Destiny 2: {average_quality_score:.2f}")
        else:
            logging.info("Keine Daten für Destiny 2 gefunden.")


    # Berechnen des Qualitätsscores
    logging.debug("Berechne Qualitätsscores:")
    df['quality_score'] = df.apply(calculate_quality_score, axis=1)

    logging.debug("Berechnete Qualitätsscores:")
    for index, row in df.iterrows():
        logging.debug(f"Title: {row['title']}")
        logging.debug(f"Quality Score: {row['quality_score']}")

    # Encoder für kategorische Variablen
    le_title = LabelEncoder()
    le_gpu_manufacturer, le_gpu_model = LabelEncoder(), LabelEncoder()
    le_distribution = LabelEncoder()
    le_cpu = LabelEncoder()
    le_ram = LabelEncoder()
    le_kernel = LabelEncoder()

    df['title_encoded'] = le_title.fit_transform(df['title'])
    df['gpu_manufacturer'], df['gpu_model'] = zip(*df['gpu'].apply(extract_gpu_info))
    df['gpu_manufacturer_encoded'] = le_gpu_manufacturer.fit_transform(df['gpu_manufacturer'])
    df['gpu_model_encoded'] = le_gpu_model.fit_transform(df['gpu_model'])
    df['distribution_encoded'] = le_distribution.fit_transform(df['distribution'])
    df['cpu_encoded'] = le_cpu.fit_transform(df['cpu'])
    df['ram_encoded'] = le_ram.fit_transform(df['ram'])
    df['kernel_encoded'] = le_kernel.fit_transform(df['kernel'])

    # Features und Zielvariable mit Gewichtung
    X = np.column_stack((
        df['title_encoded'] * 5,
        df['gpu_manufacturer_encoded'] * 2,
        df['gpu_model_encoded'] * 0.5,
        df['distribution_encoded'] * 0.5,
        df['cpu_encoded'] * 0.5,
        df['ram_encoded'] * 0.5,
        df['kernel_encoded'] * 0.5
    ))
    y = df['quality_score'].values

    # Normalisierung der Eingabedaten
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler

def create_and_train_model(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2,
                        callbacks=[early_stopping], verbose=1)

    test_loss, test_mae = model.evaluate(X_test, y_test)
    logging.debug(f"Test MAE: {test_mae}")

    return model, history

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_model_and_encoders(model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler):
    if not os.path.exists('saved_model'):
        os.makedirs('saved_model')

    model.save('saved_model/my_model.keras')
    joblib.dump(le_title, 'saved_model/le_title.joblib')
    joblib.dump(le_gpu_manufacturer, 'saved_model/le_gpu_manufacturer.joblib')
    joblib.dump(le_gpu_model, 'saved_model/le_gpu_model.joblib')
    joblib.dump(le_distribution, 'saved_model/le_distribution.joblib')
    joblib.dump(le_cpu, 'saved_model/le_cpu.joblib')
    joblib.dump(le_ram, 'saved_model/le_ram.joblib')
    joblib.dump(le_kernel, 'saved_model/le_kernel.joblib')
    joblib.dump(scaler, 'saved_model/scaler.joblib')
    logging.debug("Modell und LabelEncoder-Objekte wurden gespeichert.")

def load_model_and_encoders():
    loaded_model = tf.keras.models.load_model('saved_model/my_model.keras')
    le_title = joblib.load('saved_model/le_title.joblib')
    le_gpu_manufacturer = joblib.load('saved_model/le_gpu_manufacturer.joblib')
    le_gpu_model = joblib.load('saved_model/le_gpu_model.joblib')
    le_distribution = joblib.load('saved_model/le_distribution.joblib')
    le_cpu = joblib.load('saved_model/le_cpu.joblib')
    le_ram = joblib.load('saved_model/le_ram.joblib')
    le_kernel = joblib.load('saved_model/le_kernel.joblib')
    scaler = joblib.load('saved_model/scaler.joblib')
    return loaded_model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler

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

    # Angepasste Interpretation des Scores
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

def find_latest_archive(directory):
    pattern = os.path.join(directory, "reports_*.tar.gz")
    archives = glob.glob(pattern)
    if not archives:
        raise FileNotFoundError(f"Keine passenden Archive in {directory} gefunden.")

    def parse_date(filename):
       basename = os.path.basename(filename)
       match = re.search(r'reports_(\w+)(\d+)_(\d+)', basename)
       if not match:
           raise ValueError(f"Konnte kein Datum aus dem Dateinamen extrahieren: {basename}")

       month, day, year = match.groups()

       # Konvertiere den Monatsnamen in eine Zahl
       month_num = datetime.strptime(month, '%b').month

       # Erstelle ein vollständiges Datum
       date_str = f"{year}-{month_num:02d}-{int(day):02d}"
       return datetime.strptime(date_str, "%Y-%m-%d")

    return max(archives, key=parse_date)

def extract_archive(archive_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmpdir)

        json_file = find_json_file(tmpdir)

        if json_file:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            return json_data
        else:
            raise FileNotFoundError(f"Keine JSON-Datei im extrahierten Archiv gefunden: {archive_path}")

def find_json_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                # Optionale Validierung: Prüfen Sie, ob es sich um eine gültige JSON-Datei handelt
                if is_valid_json(json_path):
                    return json_path
    return None

def is_valid_json(file_path):
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError:
        return False

if __name__ == "__main__":
    #data_file = 'mock_data.json'
    #data_file = "/home/alex/workspace/own/caniplayonlinux/protondb-data/reports/reports_sep1_2024/reports_piiremoved.json"
    data_directory = "/home/alex/workspace/own/caniplayonlinux/protondb-data/reports/"

    try:
        if not os.path.exists('saved_model/my_model.keras'):
            logging.info("Kein gespeichertes Modell gefunden. Erstelle und trainiere neues Modell...")

            latest_archive = find_latest_archive(data_directory)
            logging.info(f"Neuestes Archiv gefunden: {latest_archive}")

            json_data = extract_archive(latest_archive)
            logging.info(f"JSON-Daten erfolgreich extrahiert und geladen")

            logging.debug("Bereite Daten vor...")
            result = prepare_data(json_data)
            if result is None:
                logging.error("Fehler beim Laden der Daten. Beende Programm.")
                exit(1)
            X_train, X_test, y_train, y_test, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler = result

            logging.debug("Qualitätsscores:")
            for score in y_train:
                logging.debug(score)

            model, history = create_and_train_model(X_train, y_train, X_test, y_test)
            plot_training_history(history)
            save_model_and_encoders(model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler)
        else:
            logging.info("Gespeichertes Modell gefunden. Lade Modell...")
            model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler = load_model_and_encoders()
            logging.info("Modell und LabelEncoder-Objekte wurden geladen.")

        # Beispielvorhersagen
        logging.info("Beispielvorhersagen:")
        predictions = [
            {
                "title": "Brewmaster: Beer Brewing Simulator",
                "gpu": "NVIDIA GeForce RTX 4070",
                "distribution": "Debian GNU/Linux 12 (bookworm)",
                "cpu": "AMD Ryzen 5 5600X 6-Core",
                "ram": "32 GB",
                "kernel": "6.1.0-21-amd64"
            },
            {
                "title": "Brewmaster: Beer Brewing Simulator",
                "gpu": "NVIDIA",
                "distribution": None,
                "cpu": None,
                "ram": None,
                "kernel": None
            },
            {
                "title": "Cyberpunk 2077",
                "gpu": "NVIDIA GeForce RTX 2080 Ti",
                "distribution": "Arch Linux",
                "cpu": "AMD Ryzen 9 5950X 16-Core",
                "ram": "32 GB",
                "kernel": "6.10.10-arch1-1"
            },
            {
                "title": "Destiny 2",
                "gpu": "NVIDIA GeForce RTX 2080 Ti",
                "distribution": "Arch Linux",
                "cpu": "AMD Ryzen 9 5950X 16-Core",
                "ram": "32 GB",
                "kernel": "6.10.10-arch1-1"
            }
        ]

        # Führen Sie die Vorhersagen durch und speichern Sie die Ergebnisse
        results = []
        for pred in predictions:
            result = predict_compatibility(
                pred["title"],
                pred["gpu"],
                pred["distribution"],
                pred["cpu"],
                pred["ram"],
                pred["kernel"],
                model, le_title, le_gpu_manufacturer, le_gpu_model, le_distribution, le_cpu, le_ram, le_kernel, scaler
            )
            results.append(result)

        # Geben Sie alle Ergebnisse aus
        for result in results:
            print(f"\nVorhersage für: {result['title']}")
            print(f"Kompatibilitätslevel: {result['compatibility_level']}")
            print(f"Qualitätsscore: {result['quality_score']:.2f}")
            if result['specs']:
                print(f"Spezifikationen: {', '.join(result['specs'])}")
            if result['unknown_labels']:
                print(f"Unbekannte Labels: {', '.join(result['unknown_labels'])}")
            if result['partial_info']:
                print(f"Teilweise Informationen: {', '.join(result['partial_info'])}")
            print("-" * 50)

    except Exception as e:
        logging.error(f"Ein Fehler ist aufgetreten: {e}")
        exit(1)
