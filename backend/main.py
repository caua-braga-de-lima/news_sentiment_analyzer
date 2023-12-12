from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.middleware.cors import CORSMiddleware
from googletrans import Translator
import joblib

app = FastAPI()

class Modelo(BaseModel):
    texto: str

# Carregar o vetorizador e o modelo
vectorizer = joblib.load('../vetorizador.pkl')
modelo = joblib.load('../modelo_treinado.pkl')
tradutor = Translator()


@app.post("/resultado")
async def mandarResposta(frase: Modelo):
    # Transformar o texto usando o vetorizador
    traduzida = tradutor.translate(frase.texto,dest='en')
    vetorizada = vectorizer.transform([traduzida.text])

    # Fazer a previsão com o modelo usando os dados vetorizados
    resultado = modelo.predict(vetorizada)

    # Retornar o resultado como JSON (convertendo para um tipo serializável)
    return {"resultado": resultado.tolist(),"Traduzido":traduzida.text}  # Convertendo para lista no exemplo do 'resultado'

origins = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:5500",  # Adicione a URL do seu frontend aqui
    "http://localhost:5500"   # Adicione a URL do seu frontend aqui
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)