from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from typing import List, Dict
import plotly.graph_objects as go
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .bert_visualization import BertVisualizer

# Inicializar el visualizador BERT (añadir con las otras inicializaciones)
bert_visualizer = BertVisualizer()

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

def calculate_cosine_angles(embeddings_data: Dict):
    """Calcula los ángulos del coseno entre el vector principal y los citados."""
    main_embedding = np.array(embeddings_data['main_patent']['embedding'])
    main_embedding = main_embedding / np.linalg.norm(main_embedding)
    
    reference_vector = np.array([1, 0])
    angles = []
    
    for patent in embeddings_data['cited_patents']:
        cited_embedding = np.array(patent['embedding'])
        cited_embedding = cited_embedding / np.linalg.norm(cited_embedding)
        cos_sim = np.dot(main_embedding, cited_embedding)
        # Restringir el valor entre -1 y 1 para evitar errores numéricos
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        angle = np.arccos(cos_sim)
        angles.append({
            'id': patent['id'],
            'angle': angle
        })
    
    return angles

def generate_cosine_plot(embeddings_data: Dict):
    """Genera el gráfico de distancia coseno con información detallada en el hover."""
    angles = calculate_cosine_angles(embeddings_data)
    
    # Crear el gráfico base
    fig = go.Figure()
    
    # Vector principal (1,0)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 0],
        mode='lines+markers',
        name='Vector Principal',
        line=dict(color='red', width=3),
        hoverinfo='text',
        text=[f'Patente Principal<br>ID: {embeddings_data["main_patent"]["id"]}', 
              f'Patente Principal<br>ID: {embeddings_data["main_patent"]["id"]}'],
    ))
    
    # Vectores citados
    for angle_data in angles:
        x = np.cos(angle_data['angle'])
        y = np.sin(angle_data['angle'])
        cosine_similarity = np.cos(angle_data['angle'])  # Calcular similitud coseno
        
        # Crear texto hover para inicio y fin del vector
        hover_text = [
            f'Patente Citada<br>' +
            f'ID: {angle_data["id"]}<br>' +
            f'Similitud Coseno: {cosine_similarity:.4f}<br>' +
            f'Ángulo: {np.degrees(angle_data["angle"]):.2f}°',
            
            f'Patente Citada<br>' +
            f'ID: {angle_data["id"]}<br>' +
            f'Similitud Coseno: {cosine_similarity:.4f}<br>' +
            f'Ángulo: {np.degrees(angle_data["angle"]):.2f}°'
        ]
        
        fig.add_trace(go.Scatter(
            x=[0, x],
            y=[0, y],
            mode='lines+markers',
            name=f'Patent {angle_data["id"]}',
            hoverinfo='text',
            text=hover_text,
            marker=dict(size=8),
            line=dict(width=2)
        ))
    
    # Configuración del layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            range=[-1.2, 1.2], 
            title='X',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgrey'
        ),
        yaxis=dict(
            range=[-1.2, 1.2], 
            title='Y', 
            scaleanchor='x', 
            scaleratio=1,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgrey'
        ),
        title='Distancia Coseno entre Patentes',
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    return fig.to_json()

def calculate_euclidean_distances(embeddings_data: Dict):
    """Calcula las distancias euclidianas entre el vector principal y los citados."""
    main_point = np.array(embeddings_data['main_patent']['reduced_embedding'])
    distances = []
    
    for patent in embeddings_data['cited_patents']:
        cited_point = np.array(patent['reduced_embedding'])
        distance = np.linalg.norm(main_point - cited_point)
        distances.append({
            'id': patent['id'],
            'distance': distance
        })
    
    return distances

def generate_euclidean_plot(embeddings_data: Dict):
    """Genera el gráfico 3D de distancia euclidiana."""
    main_point = embeddings_data['main_patent']['reduced_embedding']
    distances = calculate_euclidean_distances(embeddings_data)
    
    fig = go.Figure()
    
    # Punto principal
    fig.add_trace(go.Scatter3d(
        x=[main_point[0]],
        y=[main_point[1]],
        z=[main_point[2]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Patente Principal',
        text=[embeddings_data['main_patent']['id']],
        hoverinfo='text'
    ))
    
    # Puntos citados y líneas
    for patent, distance in zip(embeddings_data['cited_patents'], distances):
        cited_point = patent['reduced_embedding']
        
        # Línea de conexión
        fig.add_trace(go.Scatter3d(
            x=[main_point[0], cited_point[0]],
            y=[main_point[1], cited_point[1]],
            z=[main_point[2], cited_point[2]],
            mode='lines',
            line=dict(color='blue', width=2),
            hoverinfo='text',
            text=[f'Distancia: {distance["distance"]:.4f}'],
        ))
        
        # Punto citado
        fig.add_trace(go.Scatter3d(
            x=[cited_point[0]],
            y=[cited_point[1]],
            z=[cited_point[2]],
            mode='markers',
            marker=dict(size=8, color='blue'),
            name=f'Patent {patent["id"]}',
            text=[patent['id']],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        title='Distancias Euclidianas entre Patentes'
    )
    
    return fig.to_json()


@router.post("/semantic")
async def get_semantic_visualization(data: dict):
    """Endpoint para visualización semántica."""
    try:
        # Log de los datos recibidos
        logger.debug(f"Datos recibidos: {data}")

        # Validar los datos de entrada
        main_text = data.get('main_text')
        cited_text = data.get('cited_text')

        if not main_text or not cited_text:
            raise HTTPException(
                status_code=400,
                detail="Faltan campos requeridos: main_text o cited_text"
            )

        # Log de validación
        logger.debug(f"Longitud texto principal: {len(main_text)}")
        logger.debug(f"Longitud texto citado: {len(cited_text)}")

        # Procesar los textos
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=100,
            ngram_range=(1, 2)
        )

        # Ajustar y transformar los textos
        try:
            tfidf_matrix = vectorizer.fit_transform([main_text, cited_text])
            logger.debug("Matriz TF-IDF generada exitosamente")
        except Exception as e:
            logger.error(f"Error en TF-IDF: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Error al procesar los textos: {str(e)}"
            )

        # Calcular similitud coseno
        similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        logger.debug(f"Similitud calculada: {similarity}")

        # Obtener términos relevantes
        feature_names = vectorizer.get_feature_names_out()
        main_weights = tfidf_matrix[0].toarray()[0]
        cited_weights = tfidf_matrix[1].toarray()[0]

        # Preparar términos
        main_terms = sorted(
            [(term, float(score)) for term, score in zip(feature_names, main_weights) if score > 0],
            key=lambda x: x[1],
            reverse=True
        )[:50]

        cited_terms = sorted(
            [(term, float(score)) for term, score in zip(feature_names, cited_weights) if score > 0],
            key=lambda x: x[1],
            reverse=True
        )[:50]

        # Preparar la respuesta
        result = {
            "similarity": similarity,
            "main_terms": [
                {"token": term, "score": score} for term, score in main_terms
            ],
            "cited_terms": [
                {"token": term, "score": score} for term, score in cited_terms
            ]
        }

        logger.debug("Respuesta preparada exitosamente")
        return result

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
'''
@router.post("/bert")
async def get_bert_visualization(data: dict):
    """Endpoint para visualización basada en BERT."""
    try:
        main_text = data.get('main_text')
        cited_text = data.get('cited_text')

        if not main_text or not cited_text:
            raise HTTPException(
                status_code=400,
                detail="Faltan textos requeridos"
            )

        result = bert_visualizer.process_texts(main_text, cited_text)
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=400,
                detail=result['message']
            )
            
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )    
'''            

@router.post("/bert")
async def get_bert_visualization(data: dict):
    """Endpoint para visualización basada en BERT."""
    try:
        logger.info("Recibiendo solicitud de visualización BERT")
        
        main_text = data.get('main_text')
        cited_text = data.get('cited_text')

        if not main_text or not cited_text:
            raise HTTPException(
                status_code=400,
                detail="Faltan textos requeridos"
            )
            
        logger.info(f"Procesando textos para visualización BERT - Longitudes: {len(main_text)}, {len(cited_text)}")
        
        result = bert_visualizer.process_texts(main_text, cited_text)
        
        if result['status'] == 'error':
            logger.error(f"Error en procesamiento BERT: {result['message']}")
            raise HTTPException(
                status_code=400,
                detail=result['message']
            )
            
        logger.info("Visualización BERT completada exitosamente")
        return result

    except HTTPException as he:
        logger.error(f"Error HTTP en visualización BERT: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Error inesperado en visualización BERT: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )
   
