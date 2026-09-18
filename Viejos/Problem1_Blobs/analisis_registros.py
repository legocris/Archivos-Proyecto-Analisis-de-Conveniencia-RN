import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Límite de iteraciones: se descartan los registros que lo superen
# y además fija el tope del eje Y en la gráfica de eficiencia.
MAX_ITERACIONES = 100_000

# ==========================================
# 1. PARSER DE LOGS
# ==========================================

def parse_log_file(filepath):
    data = {}
    # Valores por defecto para evitar errores si faltan en el log
    data['Parada'] = 0.0 
    data['Costo_train'] = 999.9
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("Costos:"):
                    break
                
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        if "." in value:
                            data[key] = float(value)
                        else:
                            data[key] = int(value)
                    except ValueError:
                        # Entrenamientos que divergieron escriben nan/-nan(ind)/inf.
                        # Los guardamos como NaN para que pandas los ignore en las medias.
                        if value.lower().lstrip("-+").startswith(("nan", "inf")):
                            data[key] = float("nan")
                        else:
                            data[key] = value
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return None
    return data

def analyze_architecture_string(arch_str):
    parts = [int(x) for x in arch_str.strip('_').split('_') if x.isdigit()]
    if len(parts) < 2: return None

    hidden_layers = parts[1:-1]
    total_hidden_neurons = sum(hidden_layers)
    is_deep = len(hidden_layers) > 1
    
    return {
        "hidden_layers": hidden_layers,
        "total_neurons": total_hidden_neurons,
        "is_deep": is_deep,
        "tipo": "Profunda" if is_deep else "Angosta"
    }

# ==========================================
# 2. PROCESAMIENTO
# ==========================================

def process_logs_folder(folder_path):
    all_records = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    print(f"Procesando {len(files)} archivos...")
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        record = parse_log_file(filepath)
        
        if record:
            record['filename'] = filename
            arch_str = record.get('Arquitectura', '')
            arch_info = analyze_architecture_string(arch_str)
            
            if arch_info:
                record.update(arch_info)
                
                # DETERMINAR SI CONVERGIÓ (TERMINÓ DE ENTRENAR)
                # Criterio: Si el costo de entrenamiento es menor o igual a la parada
                # O si las iteraciones son significativamente menores al limite (aunque usaremos costo)
                c_train = record.get('Costo_train', 999)
                parada = record.get('Parada', 0)
                # Margen de error pequeño por punto flotante
                converged = c_train <= (parada + 0.000001)
                
                record['Convergio'] = "Si" if converged else "No"
                all_records.append(record)

    return pd.DataFrame(all_records)

# ==========================================
# 3. GENERACIÓN DE ESTADÍSTICOS Y EXCEL
# ==========================================

def save_excel_and_stats(df_subset, n_neurons, output_folder):
    filename = f"Comparativa_{n_neurons}_Neuronas.xlsx"
    path = os.path.join(output_folder, filename)
    
    # 1. Calcular Estadísticos
    stats = df_subset.groupby('Arquitectura').agg({
        'F1_test': ['mean', 'std', 'max'],
        'NIteraciones': ['mean', 'min', 'max'],
        'Convergio': lambda x: (x == 'Si').sum() # Cantidad de éxitos
    }).reset_index()
    
    # Aplanar nombres de columnas
    stats.columns = ['Arquitectura', 'F1_Promedio', 'F1_Std', 'F1_Max', 
                     'Iter_Promedio', 'Iter_Min', 'Iter_Max', 'Cant_Convergio']

    # 2. Guardar en Excel con múltiples hojas
    with pd.ExcelWriter(path) as writer:
        df_subset.to_excel(writer, sheet_name='Datos_Crudos', index=False)
        stats.to_excel(writer, sheet_name='Estadisticos', index=False)
        
    print(f"   -> Excel guardado: {filename}")
    return stats

# ==========================================
# 4. GRAFICACIÓN
# ==========================================

def add_stat_annotation(ax, df_subset, column):
    """ Agrega texto con Promedio y Std en las esquinas inferiores.
        Primera arquitectura -> esquina inferior izquierda.
        Última arquitectura  -> esquina inferior derecha. """
    # Calcular stats por arquitectura para poner texto
    stats = df_subset.groupby('Arquitectura')[column].agg(['mean', 'std']).reset_index()

    # Obtener el orden de las cajas en el eje X
    x_labels = [item.get_text() for item in ax.get_xticklabels()]

    for i, label in enumerate(x_labels):
        row = stats[stats['Arquitectura'] == label]
        if row.empty:
            continue
        mu = row['mean'].values[0]
        sigma = row['std'].values[0]
        txt = f"μ={mu:.2f}\nσ={sigma:.2f}"

        # Esquina: la primera caja abajo-izquierda, la última abajo-derecha;
        # cualquier caja intermedia queda centrada sobre su posición.
        if i == 0:
            x_pos, ha = 0.02, 'left'
        elif i == len(x_labels) - 1:
            x_pos, ha = 0.98, 'right'
        else:
            x_pos, ha = (i + 0.5) / len(x_labels), 'center'

        ax.text(x_pos, 0.02, txt, transform=ax.transAxes,
                horizontalalignment=ha, verticalalignment='bottom',
                size='small', color='black', weight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

def plot_comparison(df_subset, n_neurons, output_folder):
    # 1. ORDENAMIENTO: Angostas (False) primero, luego Profundas (True)
    df_subset = df_subset.sort_values(by=['is_deep', 'Arquitectura'])
    order_arch = df_subset['Arquitectura'].unique()
    
    # Configurar estilo
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Análisis: {n_neurons} Neuronas Totales {os.path.basename(os.getcwd())}', fontsize=16)

    # Colores base para las cajas (Azul Angosta, Rojo Profunda)
    # Creamos un diccionario de paleta basado en la columna 'tipo'
    palette_box = {"Angosta": "#3498db", "Profunda": "#e74c3c"}

    # --- GRÁFICA 1: CALIDAD (F1 TEST) ---
    # Si ninguna corrida del grupo produjo F1 válido (p.ej. todas divergieron)
    # no hay nada que graficar: evitamos que seaborn reviente.
    hay_f1 = df_subset['F1_test'].notna().any()
    if hay_f1:
        sns.boxplot(x="Arquitectura", y="F1_test", data=df_subset, ax=axes[0],
                    hue="tipo", palette=palette_box, order=order_arch, dodge=False, showfliers=False)
        sns.boxplot(x="Arquitectura", y="F1_train", data=df_subset, ax=axes[0],
                    hue="tipo", palette=palette_box, order=order_arch, dodge=False, showfliers=False, boxprops=dict(alpha=0.3))

        # Puntos individuales: Amarillo si convergió, Negro si no
        sns.stripplot(x="Arquitectura", y="F1_test", data=df_subset, ax=axes[0],
                      hue="Convergio", palette={"Si": "yellow", "No": "#333333"},
                      order=order_arch, size=3, jitter=True, edgecolor="black", linewidth=0.4, marker="o")
        add_stat_annotation(axes[0], df_subset, 'F1_test')
    else:
        axes[0].text(0.5, 0.5, "Sin datos válidos de F1\n(todas las corridas divergieron)",
                     transform=axes[0].transAxes, ha='center', va='center', size='large', color='gray')

    axes[0].set_title('Calidad: F1 Score en Test')
    axes[0].set_ylabel('F1 Score')
    
    # --- GRÁFICA 2: EFICIENCIA (ITERACIONES) ---
    sns.boxplot(x="Arquitectura", y="NIteraciones", data=df_subset, ax=axes[1],
                hue="tipo", palette=palette_box, order=order_arch, dodge=False, showfliers=False)
    
    sns.stripplot(x="Arquitectura", y="NIteraciones", data=df_subset, ax=axes[1],
                  hue="Convergio", palette={"Si": "yellow", "No": "#333333"},
                  order=order_arch, size=3, jitter=True, edgecolor="black", linewidth=0.4, marker="o")

    axes[1].set_title('Eficiencia: Cantidad de Iteraciones')
    axes[1].set_ylabel('Iteraciones (Menos es mejor)')
    # 5% de aire extra para que los puntos que tocan el límite quepan completos
    axes[1].set_ylim(0, MAX_ITERACIONES * 1.05)
    add_stat_annotation(axes[1], df_subset, 'NIteraciones')

    # Ajustar leyendas
    # Solo mostramos la leyenda de Convergencia en el segundo plot para limpiar
    handles, labels = axes[0].get_legend_handles_labels()
    # La leyenda automática mezcla hue de boxplot y stripplot, la limpiamos:
    axes[0].legend([],[], frameon=False) 
    axes[1].legend(title="¿Terminó?", loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar Gráfica
    save_path = os.path.join(output_folder, f"Grafica_{n_neurons}_Neuronas.png")
    plt.savefig(save_path)
    #plt.show()
    print(f"   -> Gráfica guardada: {save_path}")
    plt.close() # Cerrar para liberar memoria

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # RUTA DE ENTRADA
    PATH_REG = "./Reg"  # <--- AJUSTA ESTO SI ES NECESARIO
    
    
    # RUTA DE SALIDA (RESULTADOS)
    PATH_RES = os.path.join(os.path.dirname(PATH_REG), "Res")
    
    if not os.path.exists(PATH_REG):
        print(f"Error: No existe la carpeta {PATH_REG}")
    else:
        # Crear carpeta de resultados
        if not os.path.exists(PATH_RES):
            os.makedirs(PATH_RES)
            print(f"Carpeta creada: {PATH_RES}")

        # 1. Procesar Datos
        df = process_logs_folder(PATH_REG)

        # 1b. Descartar corridas que superaron el límite de iteraciones
        if not df.empty:
            n_antes = len(df)
            df = df[df['NIteraciones'] <= MAX_ITERACIONES].copy()
            descartados = n_antes - len(df)
            if descartados > 0:
                print(f"Descartados {descartados} registros con más de {MAX_ITERACIONES} iteraciones.")

        if not df.empty:
            # 2. Agrupar por cantidad de neuronas totales
            unique_neurons = df['total_neurons'].unique()
            unique_neurons.sort()
            
            print(f"\nGenerando reportes en: {PATH_RES}")
            
            for n in unique_neurons:
                subset = df[df['total_neurons'] == n]
                
                # Verificar que existan datos suficientes para comparar
                types_present = subset['tipo'].unique()
                
                print(f"\n--- Procesando grupo: {n} Neuronas Totales ---")
                
                if len(types_present) > 1:
                    # Caso ideal: Hay Angostas y Profundas para comparar
                    save_excel_and_stats(subset, n, PATH_RES)
                    plot_comparison(subset, n, PATH_RES)
                else:
                    # Solo hay de un tipo, generamos excel pero avisamos
                    print(f"   Aviso: Solo se encontraron redes tipo {types_present} para {n} neuronas.")
                    save_excel_and_stats(subset, n, PATH_RES)
                    plot_comparison(subset, n, PATH_RES) # Graficamos igual aunque sea una sola columna
                    
            print("\n¡Proceso Finalizado Exitosamente!")
        else:
            print("No se encontraron registros válidos (.txt) en la carpeta.")