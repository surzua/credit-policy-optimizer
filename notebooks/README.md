# Credit Policy Optimizer — Notebooks de Experimentación

Esta carpeta contiene la suite de experimentación para análisis de datos, benchmarking de modelos de riesgo, calibración de probabilidades, optimización financiera de políticas y auditoría de robustez y equidad.

## Estructura de la Suite

| Notebook | Título | Descripción y Objetivos |
| :--- | :--- | :--- |
| **`01_exploratory_data_analysis.ipynb`** | *Análisis Exploratorio y Calidad de Cartera* | Perfilamiento de datos crediticios, análisis univariado/bivariado, cálculo de **Weight of Evidence (WoE)** e **Information Value (IV)**, multicolinealidad. |
| **`02_model_benchmarking_and_calibration.ipynb`** | *Modelado, Benchmark y Calibración* | Comparación entre Regresión Logística (Scorecard), Random Forest y LightGBM. Métricas bancarias (**Gini**, **Kolmogorov-Smirnov KS**, ROC-AUC), curvas de fiabilidad y explicabilidad con **SHAP**. |
| **`03_economic_policy_optimization.ipynb`** | *Optimización de Políticas y P&L* | Contraste entre puntos de corte ML y el óptimo económico $p^*$. Curvas de P&L acumulado, ROE, optimización con restricciones de mora y asignación de límites por riesgo. |
| **`04_stress_testing_and_macro_scenarios.ipynb`** | *Stress Testing y Escenarios Macroeconómicos* | Resiliencia de la política ante choques de costo de fondos ($c$), incremento de severidad ($\text{LGD}$) y shifts de riesgo de crédito en recesión. Cálculo de margen de seguridad. |
| **`05_reject_inference_and_fairness_audit.ipynb`** | *Reject Inference y Auditoría de Equidad* | Simulación del sesgo de selección en carteras originadas (clientes rechazados no observados) y auditoría de métricas de equidad (Fairness & Disparate Impact). |

## Cómo Ejecutar los Notebooks

Asegúrate de tener las dependencias sincronizadas con `uv`:

```bash
# 1. Instalar dependencias completas incluyendo ipykernel y visualización
make install

# 2. Registrar el kernel en Jupyter (opcional si usas VS Code o Cursor)
uv run python -m ipykernel install --user --name credit-policy-optimizer --display-name "Python (credit-policy-optimizer)"

# 3. Abrir directamente cualquier archivo .ipynb en tu IDE o lanzar Jupyter
uv run python -m jupyterlab
```
