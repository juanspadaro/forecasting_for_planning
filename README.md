
# Forecasting for planning

**Forecast de demanda**

Este repositorio contiene los diferentes enfoques utilizados para realizar forecasting de demanda en el contexto de Facturapp, utilizando varias técnicas de modelado estadístico y machine learning. El objetivo es comparar diferentes modelos y herramientas para predecir la demanda y mejorar la eficiencia del negocio.

## Estructura del repositorio

- `.github/workflows/`: Contiene la configuración de los flujos de trabajo automatizados de GitHub Actions para despliegue y CI/CD.
  
- `01_ARIMA/`: Incluye los scripts y notebooks relacionados con el modelo ARIMA para forecasting.

- `02_Random_Forest/`: En proceso.

- `03_XGBoost/`: Contiene los scripts y experimentos realizados con XGBoost para la predicción de la demanda, buscando optimizar el rendimiento de este algoritmo de boosting.

- `04_Prophet/`: Implementaciones del modelo Prophet de Facebook con búsquedas en grid y análisis de importancia de features, comparando diferentes configuraciones del modelo.

- `05_TimeGPT/`: Exploración inicial de TimeGPT para predicción de series temporales. Se están probando los primeros pasos y comparando con otras herramientas.

- `Testing models/`: Scripts y notebooks para evaluar los modelos en función de la métrica RMSE (Root Mean Squared Error) u otras métricas.

- `csv/`: Datos de entrada procesados desde la conexión con la base de datos.

- `.gitignore`: Archivo que define qué archivos y carpetas serán ignorados por Git.

- `Cluster_model.ipynb`: Notebook donde se prueban diferentes modelos de clustering en combinación con Prophet.

- `OpenWeatherMap.ipynb`: Notebook con datos de series de temperatura de Uruguay, posiblemente para evaluar si influyen en el modelo de demanda.

- `README.md`: Archivo de documentación inicial.

## Modelos y técnicas utilizadas

1. **ARIMA (AutoRegressive Integrated Moving Average)**:
   - Modelo estadístico que captura las dependencias lineales en los datos de series temporales.
   - Utilizado como baseline en este proyecto para comparaciones.

2. **XGBoost**:
   - Modelo basado en árboles de decisión con boosting. Optimiza la predicción de demanda utilizando técnicas avanzadas de machine learning.
   
3. **Prophet**:
   - Modelo de series temporales desarrollado por Facebook. Permite modelar tanto la tendencia como las estacionalidades anuales, semanales y diarias, además de manejar regresores adicionales como feriados o eventos especiales.

4. **TimeGPT**:
   - Herramienta emergente para predicción de series temporales, que utiliza transformadores avanzados. Se está explorando su efectividad frente a los modelos más tradicionales como Prophet y ARIMA.

## Cómo ejecutar este proyecto

1. **Clona el repositorio**:

   ```bash
   git clone https://github.com/tu-usuario/forecasting_for_planning.git
   cd forecasting_for_planning
   ```

2. **Instala las dependencias**:
   
   Asegúrate de tener un entorno de Python configurado (puedes usar un entorno virtual). Instala las dependencias necesarias:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecución de notebooks**:
   
   Utiliza Jupyter Notebook o JupyterLab para abrir y ejecutar los notebooks dentro de cada carpeta de modelos (`01_ARIMA/`, `03_XGBoost/`, `04_Prophet/`, etc.). Asegúrate de tener acceso a los datos y las credenciales necesarias si se requiere conexión a una base de datos.

4. **Configuración de Prophet con feriados**:

   Si estás usando Prophet, puedes agregar días festivos del país utilizando el método `add_country_holidays`, ajustando el script en función de las necesidades locales.

5. **Evaluación de los modelos**:

   Los scripts en la carpeta `Testing models/` te permitirán evaluar y comparar el rendimiento de los modelos en función de métricas como RMSE. Puedes modificar estos scripts para ajustar la evaluación según los datos específicos de tu proyecto.
