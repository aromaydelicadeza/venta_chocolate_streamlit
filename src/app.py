import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

model = joblib.load('models/modelo_arima_chocolates.model')

st.set_page_config(
    page_title="Predicción de Ventas de Chocolate",
    layout="wide",
    page_icon="🍫"
    )

st.title("🍫 Predicción de Ventas de Chocolate")

st.sidebar.header("Configuración")

meses_a_predecir = st.sidebar.slider(
    "Número de meses a predecir",
    min_value=1,
    max_value=36,
    value=6,
    help="Selecciona el jorizonte de predicción entre 1 y 36 meses"
)

if st.sidebar.button("Generar Predicción"):
    predicciones = model.predict(n_periods=meses_a_predecir)
    fechas_futuras = pd.date_range(start='2024-02', periods=meses_a_predecir, freq='MS')
    
    df_pred = pd.DataFrame({
        'Fecha': fechas_futuras,
        'Ventas': predicciones
    })
    
    df_pred['Periodo'] = df_pred['Fecha'].dt.strftime('%b %Y')
    
    st.subheader("Resumen de la Predicción")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_ventas = df_pred['Ventas'].sum()
    promedio_mensual = df_pred['Ventas'].mean()
    idx_max = df_pred['Ventas'].idxmax()
    idx_min = df_pred['Ventas'].idxmin()
    
    col1.metric("Total de Ventas", f"${total_ventas:,.2f}")
    col2.metric("Promedio Mensual", f"${promedio_mensual:,.2f}")
    col3.metric("Mejor Mes", df_pred.loc[idx_max, 'Periodo'], f'${(df_pred.loc[idx_max, 'Ventas']-promedio_mensual):,.2f}')
    if (df_pred.loc[idx_min, 'Ventas']-promedio_mensual) < 0:
        col4.metric("Peor Mes", df_pred.loc[idx_min, 'Periodo'], f'-${abs((df_pred.loc[idx_min, 'Ventas']-promedio_mensual)):,.2f}')
    else: 
        col4.metric("Peor Mes", df_pred.loc[idx_min, 'Periodo'], f'${abs((df_pred.loc[idx_min, 'Ventas']-promedio_mensual)):,.2f}')
    
    st.subheader("Tendencia de Ventas")
    
    fig = px.line(
        df_pred,
        x='Fecha',
        y='Ventas',
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Tabla Detallada")
    
    st.dataframe(df_pred, use_container_width=True)
    csv = df_pred.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descarga Pronóstico CSV",
        data=csv,
        file_name='pronostico_ventas_chocolate.csv',
        mime='text/csv'
    )    