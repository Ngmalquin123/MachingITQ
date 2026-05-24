# Importo las librerias necesarias
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pickle
import numpy as np

# Cargamos nuestros modelos
with open("models/model.pickle", "rb") as f:
    model = pickle.load(f)

# Cargamos el archivo para estandarizar
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Creamos la API

app = FastAPI()

# Crear enlace a la carpeta de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Creamos el template que se va ejecutar cuando levantemos el servicio y entremos al backend
templates = Jinja2Templates(directory="templates")


# Crear el endpoint para el consumo de la API
# Formulario enlace
@app.get("/", response_class=HTMLResponse)
def formulario(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": None, "mensaje": None}
    )


# Implementamos una funcion para la prediccion
# Solo se piden los 8 atributos seleccionados por f_classif (ver notebook de SeleccionAtributos)
@app.post("/predict", response_class=HTMLResponse)
def predecir(
    request: Request,
    sex: float = Form(...),
    cp: float = Form(...),
    thalach: float = Form(...),
    exang: float = Form(...),
    oldpeak: float = Form(...),
    slope: float = Form(...),
    ca: float = Form(...),
    thal: float = Form(...),
):
    # Convertimos los datos a un arreglo bidimensional, manteniendo el orden exacto
    # con el que se entrenó el scaler: sex, cp, thalach, exang, oldpeak, slope, ca, thal
    datos = np.array([[sex, cp, thalach, exang, oldpeak, slope, ca, thal]])
    # Escalamos los nuevos datos
    datos_estandarizados = scaler.transform(datos)
    # Realizamos la predicción del modelo de aprendizaje
    prediccion = model.predict(datos_estandarizados)
    # Mostramos el resultado de la predicción
    resultado = int(prediccion[0])
    # Mostramos el mensaje indicado
    mensaje = ""
    # Hago condición para verificar el resultado de la predicción.
    # En este dataset (johnsmith88/heart-disease) target=0 corresponde a pacientes
    # CON enfermedad cardiaca (mayor edad, mayor oldpeak, menor thalach) y
    # target=1 corresponde a pacientes SIN enfermedad.
    if resultado == 0:
        mensaje = "La predicción indica que el paciente SÍ presenta indicios de enfermedad cardíaca."
    else:
        mensaje = "La predicción indica que el paciente NO presenta indicios de enfermedad cardíaca."
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "resultado": resultado, "mensaje": mensaje}
    )
