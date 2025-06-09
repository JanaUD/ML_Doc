![image](https://github.com/user-attachments/assets/041725b4-fce2-4ba5-9c5f-9046b7e181c7)Competición_MMLM2025_NCAA 2025

COMPETENCIA DE KAGGEL March Machine Learning Mania 2025 Forecast the 2025 NCAA Basketball Tournaments

Enlace: https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data

Enlace Colab: https://colab.research.google.com/drive/1UGjmn35SlBkCXzpmV4xXo5KZrxpWZLr6?usp=sharing

NOTA: Para ejecutar correctamente el código primero debe subir al almacenamiento de Colab el archivo .zip que contiene todas las bases de datos de la competencia.

1. Función para descomprimir el archivo .zip
      !kaggle competitions download -c march-machine-learning-mania-2025
      !unzip march-machine-learning-mania-2025.zip
2. Calculate_team_stats: Proceso para calcular los datos estadisticos relevantes de los datos representativos de cada equipo co el fin de determinar cuales son las posibles variables de peso y correlacionables para la predicción.
      def calculate_team_stats(regular_data):
        # Verificar columnas disponibles
        available_cols = regular_data.columns.tolist()
        print("Columnas disponibles:", available_cols)
    
        # Definir estadísticas básicas que intentaremos calcular
        stats_to_calculate = {
            'W': ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
                  'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'],
            'L': ['LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                  'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
        }
    
        # Filtrar solo las columnas que existen en los datos
        w_stats_cols = [col for col in stats_to_calculate['W'] if col in available_cols]
        l_stats_cols = [col for col in stats_to_calculate['L'] if col in available_cols]
    
        # Calcular estadísticas para equipos ganadores
        w_stats = regular_data.groupby(['Season', 'WTeamID'])[w_stats_cols].mean()
        w_stats = w_stats.rename(columns=lambda x: x[1:] if x.startswith('W') else x)
    
        # Calcular estadísticas para equipos perdedores
        l_stats = regular_data.groupby(['Season', 'LTeamID'])[l_stats_cols].mean()
        l_stats = l_stats.rename(columns=lambda x: x[1:] if x.startswith('L') else x) 
    
        # Combinar estadísticas
        team_stats = pd.concat([w_stats, l_stats])
        team_stats = team_stats.groupby(level=[0, 1]).mean()  # Promedio si hay duplicados
        team_stats = team_stats.reset_index()
        team_stats = team_stats.rename(columns={'WTeamID': 'TeamID'} if 'WTeamID' in team_stats.columns
                              else team_stats.rename(columns={'LTeamID': 'TeamID'}))
    
        # Calcular porcentajes solo si tenemos los datos necesarios
        if 'FGM' in team_stats.columns and 'FGA' in team_stats.columns:
            team_stats['FG%'] = team_stats['FGM'] / team_stats['FGA']
        if 'FGM3' in team_stats.columns and 'FGA3' in team_stats.columns:
            team_stats['3P%'] = team_stats['FGM3'] / team_stats['FGA3']
        if 'FTM' in team_stats.columns and 'FTA' in team_stats.columns:
            team_stats['FT%'] = team_stats['FTM'] / team_stats['FTA']
        if 'OR' in team_stats.columns and 'DR' in team_stats.columns:
            team_stats['Rebounds'] = team_stats['OR'] + team_stats['DR']
    
        return team_stats

3. Prepare_training_set: Presentación y preparación de los datos con los que se va a entrenar el modelo.
      def prepare_training_set(tourney_data, team_stats):
        if team_stats is None or len(team_stats) == 0:
            raise ValueError("team_stats está vacío o no es válido")
    
        # Verificar columnas mínimas requeridas
        required_cols = ['Season', 'WTeamID', 'LTeamID']
        missing_cols = [col for col in required_cols if col not in tourney_data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en tourney_data: {missing_cols}")
    
        # Preparar los resultados del torneo
        tourney = tourney_data[required_cols].copy()
    
        # Verificar columnas en team_stats
        if 'TeamID' not in team_stats.columns or 'Season' not in team_stats.columns:
            raise ValueError("team_stats debe contener 'TeamID' y 'Season'")
    
        # Combinar con estadísticas de equipos ganadores
        try:
            tourney = tourney.merge(team_stats,
                                   left_on=['Season', 'WTeamID'],
                                   right_on=['Season', 'TeamID'],
                                   how='left',
                                   suffixes=('', '_W'))
        except Exception as e:
            print(f"Error al fusionar estadísticas ganadoras: {str(e)}")
            return None
    
        # Combinar con estadísticas de equipos perdedores
        try:
            tourney = tourney.merge(team_stats,
                                   left_on=['Season', 'LTeamID'],
                                   right_on=['Season', 'TeamID'],
                                   how='left',
                                   suffixes=('_W', '_L'))
        except Exception as e:
            print(f"Error al fusionar estadísticas perdedoras: {str(e)}")
            return None
    
        # Crear características de diferencia entre equipos
        stat_prefixes = ['Score', 'FG', '3P', 'FT', 'Rebounds', 'Ast', 'TO', 'Stl', 'Blk']
        for stat in stat_prefixes:
            w_col = f"{stat}_W" if f"{stat}_W" in tourney.columns else stat
            l_col = f"{stat}_L" if f"{stat}_L" in tourney.columns else stat
    
            if w_col in tourney.columns and l_col in tourney.columns:
                tourney[f'{stat}_Diff'] = tourney[w_col] - tourney[l_col]
            else:
                print(f"Advertencia: No se encontraron columnas para {stat}")
    
        # La variable objetivo es 1 si el primer equipo gana
        tourney['target'] = 1
    
        # También necesitamos ejemplos donde el orden de los equipos se invierte
        reverse = tourney.copy()
        for col in tourney.columns:
            if col.endswith('_Diff'):
                reverse[col] = -reverse[col]
        reverse['target'] = 0
    
        # Combinar ambos conjuntos
        full_data = pd.concat([tourney, reverse])
    
        return full_data

3.1. Verificacion de las columnas reales en los datos:
  import os

    # Listar archivos en el directorio actual
    print("Archivos en el directorio actual:", os.listdir())
      import pandas as pd
    from IPython.display import display
    
    def load_and_display_data(gender):
        """Carga y muestra los datos para un género específico"""
        prefix = gender
        gender_name = 'Masculino' if gender == 'M' else 'Femenino'
    
        print(f"\n=== DATOS {gender_name.upper()} ===")
    
        try:
            # Cargar archivos
            regular = pd.read_csv(f'{prefix}RegularSeasonDetailedResults.csv')
            tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
            seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
    
            # Mostrar información básica
            print(f"\n1. Temporada Regular ({len(regular)} registros):")
            display(regular.head(3))
            print("\nColumnas disponibles:", regular.columns.tolist())
    
            print(f"\n2. Torneo NCAA ({len(tourney)} registros):")
            display(tourney.head(3))
            print("\nColumnas disponibles:", tourney.columns.tolist())
    
            print(f"\n3. Semillas del Torneo ({len(seeds)} registros):")
            display(seeds.head(3))
            print("\nColumnas disponibles:", seeds.columns.tolist())
    
            return regular, tourney, seeds
    
        except Exception as e:
            print(f"Error al cargar datos {gender_name.lower()}: {str(e)}")
            return None, None, None
    
    # Mostrar datos masculinos
    m_regular, m_tourney, m_seeds = load_and_display_data('M')
    
    # Mostrar datos femeninos
    w_regular, w_tourney, w_seeds = load_and_display_data('W')
    
    # Comparativa básica
    if m_regular is not None and w_regular is not None:
        print("\n=== COMPARATIVA ===")
        print(f"Partidos temporada regular: Masculino={len(m_regular)}, Femenino={len(w_regular)}")
        print(f"Partidos de torneo: Masculino={len(m_tourney)}, Femenino={len(w_tourney)}")
        print(f"Equipos con semilla: Masculino={len(m_seeds)}, Femenino={len(w_seeds)}")
RESULTADOS

=== DATOS MASCULINO ===

Temporada Regular (118882 registros):

![image](https://github.com/user-attachments/assets/30b66b35-c487-4943-9810-af862d8dd16b)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Torneo NCAA (2518 registros):

![image](https://github.com/user-attachments/assets/a868ad3d-4b2c-441d-9ac7-1096b65e59cc)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Semillas del Torneo (2626 registros):

![image](https://github.com/user-attachments/assets/34d083b1-8d89-4287-893b-4699c5926940)

Columnas disponibles: ['Season', 'Seed', 'TeamID']

=== DATOS FEMENINO ===

Temporada Regular (81708 registros):

![image](https://github.com/user-attachments/assets/00213c12-fa91-4ebc-8dee-cc0adca18ed7)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Torneo NCAA (1650 registros):

![image](https://github.com/user-attachments/assets/6ddfcdc9-19db-4c65-a433-c1d9e9b1bba7)

Columnas disponibles: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Semillas del Torneo (1744 registros):

![image](https://github.com/user-attachments/assets/18c20f25-e442-4c5a-825a-03405b6acaae)

Columnas disponibles: ['Season', 'Seed', 'TeamID']

=== COMPARATIVA === Partidos temporada regular: Masculino=118882, Femenino=81708 Partidos de torneo: Masculino=2518, Femenino=1650 Equipos con semilla: Masculino=2626, Femenino=1744

3.2. Correlación y normalización de datos para el entrenamiento:
      import pandas as pd
    from IPython.display import display, HTML
    
    def display_data_with_info(df, title):
        """Muestra un DataFrame con información detallada"""
        display(HTML(f"<h3>{title}</h3>"))
        display(df.head(3))
        print(f"\nFilas: {len(df)} | Columnas: {len(df.columns)}")
        print("Columnas:", df.columns.tolist())
        print("\nEstadísticas descriptivas:")
        display(df.describe(include='all').head(3))
    
    def analyze_gender_data(gender):
        """Analiza y muestra datos para un género"""
        prefix = gender
        gender_name = 'Masculino' if gender == 'M' else 'Femenino'
    
        print(f"\n{'='*50}")
        print(f"{' ANÁLISIS DE DATOS ' + gender_name.upper() + ' ':=^50}")
        print(f"{'='*50}")
    
        try:
            # Cargar datos
            regular = pd.read_csv(f'{prefix}RegularSeasonDetailedResults.csv')
            tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
            seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
            teams = pd.read_csv(f'{prefix}Teams.csv')
    
            # Mostrar datos
            display_data_with_info(regular, f"1. Temporada Regular {gender_name}")
            display_data_with_info(tourney, f"2. Torneo NCAA {gender_name}")
            display_data_with_info(seeds, f"3. Semillas del Torneo {gender_name}")
            display_data_with_info(teams, f"4. Equipos {gender_name}")
    
            return regular, tourney, seeds, teams
    
        except Exception as e:
            print(f"Error al cargar datos {gender_name.lower()}: {str(e)}")
            return None, None, None, None
    
    # Analizar ambos géneros
    m_data = analyze_gender_data('M')
    w_data = analyze_gender_data('W')

RESULTADOS:

ANÁLISIS DE DATOS MASCULINO

Temporada Regular Masculino

![image](https://github.com/user-attachments/assets/c756dd84-60f6-40b5-8834-a3b0748c1c34)


Filas: 118882 | Columnas: 34 Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/062fc776-fd1d-454c-aff8-e39e247715cd)


Torneo NCAA Masculino

![image](https://github.com/user-attachments/assets/ff91a729-b47f-4968-a0dc-2a431a7a8b2f)

Filas: 2518 | Columnas: 8 Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/55d8c2f1-f773-4263-9dab-ded941560b7a)

Semillas del Torneo Masculino

![image](https://github.com/user-attachments/assets/52df46de-bb3f-4cee-844d-01cb6f545be6)

Filas: 2626 | Columnas: 3 Columnas: ['Season', 'Seed', 'TeamID']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/7ebfb7d3-38b6-47f2-ac11-58fab62eed9c)


Equipos Masculino

![image](https://github.com/user-attachments/assets/39ba43d6-03e0-4582-a4e7-fb4a5e5a1a90)

Filas: 380 | Columnas: 4 Columnas: ['TeamID', 'TeamName', 'FirstD1Season', 'LastD1Season']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/daf5c6b9-820d-409f-b82d-3aa50626abbc)

ANÁLISIS DE DATOS FEMENINO

Temporada Regular Femenino

![image](https://github.com/user-attachments/assets/15643c93-e22a-4cec-ae88-5b7aead07534)


Filas: 81708 | Columnas: 34 Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

Torneo NCAA Femenino

![image](https://github.com/user-attachments/assets/7f7921a2-f2dd-45c2-aa8e-e197e3383c2c)

Filas: 1650 | Columnas: 8 Columnas: ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc', 'NumOT']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/7490b8f6-ad23-4106-b796-3b9732cf4cc6)

Semillas del Torneo Femenino

![image](https://github.com/user-attachments/assets/0124c3eb-9aa0-4ef3-b1c5-98a5b6344856)


Filas: 1744 | Columnas: 3 Columnas: ['Season', 'Seed', 'TeamID']

Estadísticas descriptivas:

![image](https://github.com/user-attachments/assets/4e5436d5-a1bf-45eb-b023-573c2676a25e)

Equipos Femenino

![image](https://github.com/user-attachments/assets/9b425cf9-cb69-4784-92f7-936faa0752a1)

Filas: 378 | Columnas: 2 Columnas: ['TeamID', 'TeamName']

Estadísticas descriptivas:

image

4. Cálculo de las estadísticascon los datos relevantes del entrenamiento del modelo:
      def calculate_team_stats(regular_data):
        # Verificar que tenemos las columnas mínimas necesarias
        required_cols = ['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']
        missing_cols = [col for col in required_cols if col not in regular_data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
    
        # Estadísticas básicas para equipos ganadores
        w_stats = regular_data.groupby(['Season', 'WTeamID']).agg({
            'WScore': 'mean',
            'LScore': 'mean',
            'WFGM': 'mean',
            'WFGA': 'mean',
            'WFGM3': 'mean',
            'WFGA3': 'mean',
            'WFTM': 'mean',
            'WFTA': 'mean',
            'WOR': 'mean',
            'WDR': 'mean',
            'WAst': 'mean',
            'WTO': 'mean',
            'WStl': 'mean',
            'WBlk': 'mean',
            'WPF': 'mean'
        }).reset_index()
    
        # Renombrar columnas (quitamos la W inicial)
        w_stats = w_stats.rename(columns={
            'WTeamID': 'TeamID',
            **{col: col[1:] for col in w_stats.columns if col.startswith('W') and col not in ['WTeamID']}
        })
    
        # Estadísticas básicas para equipos perdedores
        l_stats = regular_data.groupby(['Season', 'LTeamID']).agg({
            'LScore': 'mean',
            'WScore': 'mean',
            'LFGM': 'mean',
            'LFGA': 'mean',
            'LFGM3': 'mean',
            'LFGA3': 'mean',
            'LFTM': 'mean',
            'LFTA': 'mean',
            'LOR': 'mean',
            'LDR': 'mean',
            'LAst': 'mean',
            'LTO': 'mean',
            'LStl': 'mean',
            'LBlk': 'mean',
            'LPF': 'mean'
        }).reset_index()
    
        # Renombrar columnas (quitamos la L inicial)
        l_stats = l_stats.rename(columns={
            'LTeamID': 'TeamID',
            **{col: col[1:] for col in l_stats.columns if col.startswith('L') and col not in ['LTeamID']}
        })
    
        # Combinar estadísticas (promedio de ambas perspectivas)
        team_stats = pd.concat([w_stats, l_stats]).groupby(['Season', 'TeamID']).mean().reset_index()
    
        # Calcular métricas derivadas
        if 'FGM' in team_stats.columns and 'FGA' in team_stats.columns:
            team_stats['FG%'] = team_stats['FGM'] / team_stats['FGA']
        if 'FGM3' in team_stats.columns and 'FGA3' in team_stats.columns:
            team_stats['3P%'] = team_stats['FGM3'] / team_stats['FGA3']
        if 'FTM' in team_stats.columns and 'FTA' in team_stats.columns:
            team_stats['FT%'] = team_stats['FTM'] / team_stats['FTA']
        if 'OR' in team_stats.columns and 'DR' in team_stats.columns:
            team_stats['Rebounds'] = team_stats['OR'] + team_stats['DR']
        if 'Ast' in team_stats.columns and 'TO' in team_stats.columns:
            team_stats['Ast/TO'] = team_stats['Ast'] / team_stats['TO']
    
        return team_stats

5. Entrenamiento y resultados del modelo:
      import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from itertools import combinations
    
    def load_and_validate_data(gender):
        """Carga y valida los datos con manejo de errores mejorado"""
        try:
            prefix = gender
            print(f"\nCargando datos para el torneo {'masculino' if gender == 'M' else 'femenino'}...")
    
            # Cargar archivos con verificación
            teams = pd.read_csv(f'{prefix}Teams.csv')
            seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
            regular = pd.read_csv(f'{prefix}RegularSeasonCompactResults.csv')  # Usamos versión compacta
            tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
    
            # Verificar columnas disponibles
            print("\nColumnas en cada archivo:")
            print(f"Teams: {teams.columns.tolist()}")
            print(f"Seeds: {seeds.columns.tolist()}")
            print(f"Regular: {regular.columns.tolist()}")
            print(f"Tourney: {tourney.columns.tolist()}")
    
            # Renombrar columnas si es necesario (para compatibilidad)
            if 'WTeamID' not in regular.columns:
                if 'Winner' in regular.columns:  # Ejemplo de adaptación
                    regular = regular.rename(columns={'Winner': 'WTeamID', 'Loser': 'LTeamID'})
                else:
                    raise ValueError("No se encontraron columnas de equipos ganadores/perdedores")
    
            return teams, seeds, regular, tourney
    
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None, None, None, None
    
    def build_simple_model(regular_data):
        """Construye un modelo simple basado en semillas y puntuaciones"""
        try:
            # Calcular estadísticas básicas
            win_stats = regular_data.groupby('WTeamID')['WScore'].mean().reset_index()
            loss_stats = regular_data.groupby('LTeamID')['LScore'].mean().reset_index()
    
            team_stats = pd.merge(
                win_stats.rename(columns={'WTeamID': 'TeamID', 'WScore': 'Offense'}),
                loss_stats.rename(columns={'LTeamID': 'TeamID', 'LScore': 'Defense'}),
                on='TeamID',
                how='outer'
            ).fillna(0)
    
            return team_stats
    
        except Exception as e:
            print(f"Error construyendo modelo simple: {e}")
            return None
    
    def predict_winners(team_stats, seeds, teams, current_season=2025):
        """Predice ganadores usando un enfoque basado en semillas y estadísticas"""
        try:
            # Procesar semillas
            seeds = seeds[seeds['Season'] == current_season].copy()
            seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)
    
            # Combinar con estadísticas
            predictions = seeds.merge(
                team_stats,
                on='TeamID',
                how='left'
            ).merge(
                teams[['TeamID', 'TeamName']],
                on='TeamID',
                how='left'
            )
    
            # Si faltan estadísticas, usar solo la semilla
            predictions['Offense'] = predictions['Offense'].fillna(80 - predictions['SeedNum'])
            predictions['Defense'] = predictions['Defense'].fillna(60 + predictions['SeedNum'])
    
            return predictions[['TeamID', 'TeamName', 'Seed', 'SeedNum', 'Offense', 'Defense']]
    
        except Exception as e:
            print(f"Error generando predicciones: {e}")
            return None
    
    def simulate_round(teams_df, round_name):
        """Simula una ronda del torneo"""
        try:
            # Ordenar por semilla
            sorted_teams = teams_df.sort_values('SeedNum')
            winners = []
            results = []
    
            # Determinar emparejamientos según la ronda
            if round_name == "Ronda 1":
                # Emparejar 1 vs 16, 2 vs 15, etc.
                top = sorted_teams.iloc[:8]
                bottom = sorted_teams.iloc[8:16].iloc[::-1]
                matchups = list(zip(top.values, bottom.values))
            elif round_name == "Ronda 2":
                # Emparejar ganadores de Ronda 1
                if len(sorted_teams) == 8:
                    matchups = list(zip(sorted_teams.iloc[::2].values, sorted_teams.iloc[1::2].values))
                else:
                    matchups = []
            elif round_name == "Sweet 16":
                if len(sorted_teams) == 4:
                    matchups = list(zip(sorted_teams.iloc[::2].values, sorted_teams.iloc[1::2].values))
                else:
                    matchups = []
            elif round_name == "Elite 8":
                if len(sorted_teams) == 2:
                    matchups = [(sorted_teams.iloc[0].values, sorted_teams.iloc[1].values)]
                else:
                    matchups = []
            else:
                matchups = []
    
            for team1, team2 in matchups:
                # Calcular probabilidad simple basada en semillas y estadísticas
                prob = 0.5 + (team2[4] - team1[4])/100  # Ajuste basado en ofensa
    
                if prob >= 0.5:
                    winner = team1
                    prob = min(0.95, prob)  # Limitar probabilidad máxima
                else:
                    winner = team2
                    prob = max(0.05, 1-prob)
    
                results.append({
                    'Ronda': round_name,
                    'Equipo1': team1[1],  # TeamName
                    'Equipo2': team2[1],
                    'Probabilidad': f"{prob:.2f}",
                    'Ganador': winner[1]
                })
                winners.append(winner)
    
            # Crear DataFrames de resultados y ganadores
            results_df = pd.DataFrame(results)
            if winners:
                winners_df = pd.DataFrame(winners, columns=teams_df.columns)
            else:
                winners_df = pd.DataFrame(columns=teams_df.columns)
    
            return results_df, winners_df
    
        except Exception as e:
            print(f"Error en simulate_round: {e}")
            return pd.DataFrame(), pd.DataFrame(columns=teams_df.columns)
    
    def display_round_results(results_df):
        """Muestra los resultados de una ronda"""
        try:
            if results_df.empty:
                print("\nNo hay resultados para mostrar en esta ronda.")
                return
    
            if 'Ronda' in results_df.columns:
                print(f"\n{results_df['Ronda'].iloc[0].upper()}:")
            else:
                print("\nResultados de la ronda:")
    
            for _, row in results_df.iterrows():
                print(f"  {row['Equipo1']} vs {row['Equipo2']}")
                print(f"  Probabilidad: {row['Probabilidad']} -> GANADOR: {row['Ganador']}")
                print("  " + "-"*40)
        except Exception as e:
            print(f"Error mostrando resultados: {e}")
    
    def predict_tournament(gender):
        """Predice el torneo completo para un género"""
        try:
            # Cargar datos
            teams, seeds, regular, tourney = load_and_validate_data(gender)
            if teams is None:
                return
    
            # Construir modelo simple
            team_stats = build_simple_model(regular)
            if team_stats is None:
                return
    
            # Predecir ganadores
            predictions = predict_winners(team_stats, seeds, teams)
            if predictions is None:
                return
    
            # Simular rondas
            round_names = ["Ronda 1", "Ronda 2", "Sweet 16", "Elite 8", "Final Four", "Final"]
    
            print(f"\n{'*'*50}")
            print(f"PREDICCIONES TORNEO {'MASCULINO' if gender == 'M' else 'FEMENINO'} 2025")
            print(f"{'*'*50}")
    
            current_teams = predictions.copy()
    
            for round_name in round_names:
                if len(current_teams) < 2:
                    print(f"\nNo hay suficientes equipos para continuar ({len(current_teams)} restantes)")
                    break
    
                results, winners = simulate_round(current_teams, round_name)
                display_round_results(results)
                current_teams = winners
    
        except Exception as e:
            print(f"Error en predict_tournament: {e}")
    
    # Ejecutar predicciones
    print("PREDICCIÓN DE GANADORES POR RONDA - NCAA 2025")
    predict_tournament('M')  # Torneo masculino
    predict_tournament('W')  # Torneo femenino
    
    print("\nDesarrollado por: J.E. Carmona Alvarez & J. Ortiz-Aguilar")

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from itertools import combinations
    
    def load_and_validate_data(gender):
        """Carga y valida los datos con manejo de errores mejorado"""
        try:
            prefix = gender
            print(f"\nCargando datos para el torneo {'masculino' if gender == 'M' else 'femenino'}...")
    
            # Cargar archivos con verificación
            teams = pd.read_csv(f'{prefix}Teams.csv')
            seeds = pd.read_csv(f'{prefix}NCAATourneySeeds.csv')
            regular = pd.read_csv(f'{prefix}RegularSeasonCompactResults.csv')  # Usamos versión compacta
            tourney = pd.read_csv(f'{prefix}NCAATourneyCompactResults.csv')
    
            # Verificar columnas disponibles
            print("\nColumnas en cada archivo:")
            print(f"Teams: {teams.columns.tolist()}")
            print(f"Seeds: {seeds.columns.tolist()}")
            print(f"Regular: {regular.columns.tolist()}")
            print(f"Tourney: {tourney.columns.tolist()}")
    
            # Renombrar columnas si es necesario (para compatibilidad)
            if 'WTeamID' not in regular.columns:
                if 'Winner' in regular.columns:  # Ejemplo de adaptación
                    regular = regular.rename(columns={'Winner': 'WTeamID', 'Loser': 'LTeamID'})
                else:
                    raise ValueError("No se encontraron columnas de equipos ganadores/perdedores")
    
            return teams, seeds, regular, tourney
    
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None, None, None, None
    
    def build_simple_model(regular_data):
        """Construye un modelo simple basado en semillas y puntuaciones"""
        try:
            # Calcular estadísticas básicas
            win_stats = regular_data.groupby('WTeamID')['WScore'].mean().reset_index()
            loss_stats = regular_data.groupby('LTeamID')['LScore'].mean().reset_index()
    
            team_stats = pd.merge(
                win_stats.rename(columns={'WTeamID': 'TeamID', 'WScore': 'Offense'}),
                loss_stats.rename(columns={'LTeamID': 'TeamID', 'LScore': 'Defense'}),
                on='TeamID',
                how='outer'
            ).fillna(0)
    
            return team_stats
    
        except Exception as e:
            print(f"Error construyendo modelo simple: {e}")
            return None
    
    def predict_winners(team_stats, seeds, teams, current_season=2025):
        """Predice ganadores usando un enfoque basado en semillas y estadísticas"""
        try:
            # Procesar semillas
            seeds = seeds[seeds['Season'] == current_season].copy()
            seeds['SeedNum'] = seeds['Seed'].str.extract('(\d+)').astype(int)
    
            # Combinar con estadísticas
            predictions = seeds.merge(
                team_stats,
                on='TeamID',
                how='left'
            ).merge(
                teams[['TeamID', 'TeamName']],
                on='TeamID',
                how='left'
            )
    
            # Si faltan estadísticas, usar solo la semilla
            predictions['Offense'] = predictions['Offense'].fillna(80 - predictions['SeedNum'])
            predictions['Defense'] = predictions['Defense'].fillna(60 + predictions['SeedNum'])
    
            return predictions[['TeamID', 'TeamName', 'Seed', 'SeedNum', 'Offense', 'Defense']]
    
        except Exception as e:
            print(f"Error generando predicciones: {e}")
            return None
    
    def simulate_round(teams_df, round_name):
        """Simula una ronda del torneo"""
        try:
            # Ordenar por semilla
            sorted_teams = teams_df.sort_values('SeedNum')
            winners = []
            results = []
    
            # Determinar emparejamientos según la ronda
            if round_name == "Ronda 1":
                # Emparejar 1 vs 16, 2 vs 15, etc.
                top = sorted_teams.iloc[:8]
                bottom = sorted_teams.iloc[8:16].iloc[::-1]
                matchups = list(zip(top.values, bottom.values))
            elif round_name == "Ronda 2":
                # Emparejar ganadores de Ronda 1
                if len(sorted_teams) == 8:
                    matchups = list(zip(sorted_teams.iloc[::2].values, sorted_teams.iloc[1::2].values))
                else:
                    matchups = []
            elif round_name == "Sweet 16":
                if len(sorted_teams) == 4:
                    matchups = list(zip(sorted_teams.iloc[::2].values, sorted_teams.iloc[1::2].values))
                else:
                    matchups = []
            elif round_name == "Elite 8":
                if len(sorted_teams) == 2:
                    matchups = [(sorted_teams.iloc[0].values, sorted_teams.iloc[1].values)]
                else:
                    matchups = []
            else:
                matchups = []
    
            for team1, team2 in matchups:
                # Calcular probabilidad simple basada en semillas y estadísticas
                prob = 0.5 + (team2[4] - team1[4])/100  # Ajuste basado en ofensa
    
                if prob >= 0.5:
                    winner = team1
                    prob = min(0.95, prob)  # Limitar probabilidad máxima
                else:
                    winner = team2
                    prob = max(0.05, 1-prob)
    
                results.append({
                    'Ronda': round_name,
                    'Equipo1': team1[1],  # TeamName
                    'Equipo2': team2[1],
                    'Probabilidad': f"{prob:.2f}",
                    'Ganador': winner[1]
                })
                winners.append(winner)
    
            # Crear DataFrames de resultados y ganadores
            results_df = pd.DataFrame(results)
            if winners:
                winners_df = pd.DataFrame(winners, columns=teams_df.columns)
            else:
                winners_df = pd.DataFrame(columns=teams_df.columns)
    
            return results_df, winners_df
    
        except Exception as e:
            print(f"Error en simulate_round: {e}")
            return pd.DataFrame(), pd.DataFrame(columns=teams_df.columns)
    
    def display_round_results(results_df):
        """Muestra los resultados de una ronda"""
        try:
            if results_df.empty:
                print("\nNo hay resultados para mostrar en esta ronda.")
                return
    
            if 'Ronda' in results_df.columns:
                print(f"\n{results_df['Ronda'].iloc[0].upper()}:")
            else:
                print("\nResultados de la ronda:")
    
            for _, row in results_df.iterrows():
                print(f"  {row['Equipo1']} vs {row['Equipo2']}")
                print(f"  Probabilidad: {row['Probabilidad']} -> GANADOR: {row['Ganador']}")
                print("  " + "-"*40)
        except Exception as e:
            print(f"Error mostrando resultados: {e}")
    
    def predict_tournament(gender):
        """Predice el torneo completo para un género"""
        try:
            # Cargar datos
            teams, seeds, regular, tourney = load_and_validate_data(gender)
            if teams is None:
                return
    
            # Construir modelo simple
            team_stats = build_simple_model(regular)
            if team_stats is None:
                return
    
            # Predecir ganadores
            predictions = predict_winners(team_stats, seeds, teams)
            if predictions is None:
                return
    
            # Simular rondas
            round_names = ["Ronda 1", "Ronda 2", "Sweet 16", "Elite 8", "Final Four", "Final"]
    
            print(f"\n{'*'*50}")
            print(f"PREDICCIONES TORNEO {'MASCULINO' if gender == 'M' else 'FEMENINO'} 2025")
            print(f"{'*'*50}")
    
            current_teams = predictions.copy()
    
            for round_name in round_names:
                if len(current_teams) < 2:
                    print(f"\nNo hay suficientes equipos para continuar ({len(current_teams)} restantes)")
                    break
    
                results, winners = simulate_round(current_teams, round_name)
                display_round_results(results)
                current_teams = winners
    
        except Exception as e:
            print(f"Error en predict_tournament: {e}")
    
    # Ejecutar predicciones
    print("PREDICCIÓN DE GANADORES POR RONDA - NCAA 2025")
    predict_tournament('M')  # Torneo masculino
    predict_tournament('W')  # Torneo femenino
    
    print("\nDesarrollado por: J.E. Carmona Alvarez & J. Ortiz-Aguilar")


