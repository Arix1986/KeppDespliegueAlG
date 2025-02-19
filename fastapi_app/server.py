from typing import Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
from app_class import *
import uvicorn
from app_class import *
from pydantic import BaseModel
import mlflow

app = FastAPI()
host = "127.0.0.1" 
port = 4500

class DataToken(BaseModel):
    use_lemma: Optional[bool]=True
    use_stop_wd:Optional[bool]=True
    texto: str



@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>Predictor of Sentiments</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script>
                $(document).ready(function() {
            $("#submit_form").click(function() {
                let inputText = $("#input").val();
                 $("#ansewer").val("");
                if (!inputText) {
                    Swal.fire("Error", "Please enter a text!", "warning");
                    return;
                }

                // Mostrar alerta de carga
                Swal.fire({
                    title: "Processing...",
                    text: "Please wait while we analyze the sentiment",
                    allowOutsideClick: false,
                    didOpen: () => {
                        Swal.showLoading();
                    }
                });

               
                $.ajax({
                    url: `/items/${encodeURIComponent(inputText)}`,
                    type: "GET",
                    success: function(response) {
                        Swal.close();
                        $("#ansewer").val(response.sentiment);
                      
                    },
                    error: function(xhr) {
                        Swal.fire("Error", xhr.responseJSON.error || "Something went wrong", "error");
                    }
                });
            });
        });
            </script>
            <style>
                body { margin: 20px; font-family: Arial, sans-serif; }
                h1 { text-align: center; margin-bottom: 40px; }
                .graph-container {
                    padding: 20px; border: 1px solid #ddd; border-radius: 5px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Predictor of Sentiments</h1>
                <div class="row">
                    <div class="col-md-6">
                        <div class="form-group">
                            <label for="input">Text Input</label>
                            <input class="form-control" type="text" id="input" placeholder="Enter text">
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="form-group">
                            <label for="ansewer">Sentiment</label>
                            <input class="form-control" type="text" id="ansewer" readonly>
                        </div>
                    </div>
                    <div class="col-md-12 text-center mt-4">
                        <button type="button" id="submit_form" class="btn btn-primary">Predict</button>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """     

@app.post("/predict")
def predict(texto: str):
   if texto.strip(): 
        try:
            logged_model = 'runs:/48d306bda1274ab29cdb667943c80a30/model'
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            tokenizer=Tokenizer()
            x_test_tokenized = tokenizer.preprocess_texts([texto],1)
            x_test_joined=[" ".join(token)  for token in x_test_tokenized]
            prediction = loaded_model.predict(x_test_joined)
            if prediction[0]==0:
                return JSONResponse(content={"predict": float(prediction[0]),"sentiment":"Negative"}, status_code=200)
            else:
                return JSONResponse(content={"predict": float(prediction[0]),"sentiment":"Positive"}, status_code=200)
            
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)  
           
   else:
        return JSONResponse(content={"error": "No se ha ingresado texto"}, status_code=400)   


@app.post("/show_df_train")
def show_df(n_elements: int):
    df=pd.read_csv('./datasets/train.csv')
    return df.head(n_elements).to_dict(orient="records")



@app.post("/tokenizar_texto")
def tokenizar_texto(data: DataToken):
    token=Tokenizer(use_lemma=data.use_lemma, use_stw=data.use_stop_wd)
    texto_token=token.preprocess_texts([data.texto],1)
    x_test_joined=[" ".join(token)  for token in texto_token]
    return {'Token:', f'{x_test_joined}'}


@app.post('/predict_with_probs/')
def predict_probs(texto: str):
   if texto.strip(): 
        try:
            logged_model = 'runs:/48d306bda1274ab29cdb667943c80a30/model'
            loaded_model = mlflow.sklearn.load_model(logged_model)
            tokenizer=Tokenizer()
            x_test_tokenized = tokenizer.preprocess_texts([texto],1)
            x_test_joined=[" ".join(token)  for token in x_test_tokenized]
            prediction_proba = loaded_model.predict_proba(x_test_joined)[0]
            prob_positive = prediction_proba[1]  
            prob_negative = prediction_proba[0] 

            
            sentiment = "Positive" if prob_positive > 0.5 else "Negative"

            return JSONResponse(content={
                "sentiment": sentiment,
                "prob_positive": float(prob_positive),
                "prob_negative": float(prob_negative)
            }, status_code=200)
            
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)  
           
   else:
        return JSONResponse(content={"error": "No se ha ingresado texto"}, status_code=400)   




