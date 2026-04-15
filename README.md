# Transformer: Traductor de Inglés a Español desde Cero

[Paper](https://arxiv.org/abs/1706.03762) | [PyTorch](https://pytorch.org/) | [License: MIT](https://opensource.org/licenses/MIT)

Este repositorio contiene la implementación completa, desde cero, de un modelo Transformer basado en la arquitectura Encoder-Decoder descrita en el artículo original "Attention Is All You Need" (Vaswani et al., 2017). 

El objetivo principal de este proyecto es entrenar un modelo de Traducción Automática Neuronal (NMT) capaz de traducir texto de inglés a español.

---

## Índice
1. [Qué es este proyecto](#qué-es-este-proyecto)
2. [Arquitectura y Configuración](#arquitectura-y-configuración)
3. [Datasets de Entrenamiento](#datasets-de-entrenamiento)
4. [Tokenización y Vocabulario Custom (BPE)](#tokenización-y-vocabulario-custom-bpe)
5. [CLI: Instalación y Uso](#cli)

---

## Qué es este proyecto

Este repositorio es la implementación práctica de mi **Trabajo de Fin de Grado (TFG) en Matemáticas**. 

> **Nota:** Mientras que en la memoria escrita del TFG analizo y explico en detalle la teoría y todos los conceptos matemáticos que fundamentan esta arquitectura, este proyecto es el fruto de poner en práctica esos conocimientos.

Por esta razón, **no se han utilizado librerías de alto nivel** orientadas a NLP (como *Hugging Face*). En su lugar, el modelo ha sido programado desde cero, componente por componente (Self-Attention, Multi-Head Attention, Positional Encoding, etc.) utilizando **PyTorch**. El objetivo de este enfoque es comprender en profundidad las operaciones matriciales, el comportamiento interno y el flujo de los tensores que hacen posible el correcto funcionamiento del Transformer.

### ¿Qué es un Transformer?
Antes de 2017, la traducción automática estaba dominada por Redes Neuronales Recurrentes (RNNs) y LSTMs, las cuales procesaban el texto palabra por palabra, siendo lentas y perdiendo el contexto en frases largas. 

El **Transformer** revolucionó la Inteligencia Artificial al eliminar la recurrencia y utilizar únicamente **Mecanismos de Atención** (*Self-Attention* y *Cross-Attention*). Esto permite al modelo:
1. **Paralelización:** Procesar todas las palabras de una frase simultáneamente.
2. **Contexto Global:** Entender qué palabras de una oración están relacionadas entre sí, independientemente de la distancia que las separe.

---

## Arquitectura y Configuración

El proyecto replica la arquitectura clásica **Encoder-Decoder**. El Encoder procesa la frase en inglés y extrae su significado profundo, mientras que el Decoder toma esa información y genera la traducción al español, prestando atención a las partes relevantes del texto original paso a paso.

<p align="center">
  <img src="media/transformer.png" width="400" alt="Transformer Architecture">
  <br>
  <em>Arquitectura del Transformer original (Vaswani et al., 2017)</em>
</p>

### Parámetros del Modelo
Para este entrenamiento, el modelo ha sido instanciado con la siguiente configuración técnica:

| Parámetro | Valor | Descripción |
| :--- | :--- | :--- |
| `CONTEXT_LENGTH` | **128** | Longitud máxima de las secuencias de entrada y salida (en tokens). |
| `D_EMBEDDING` | **256** | Dimensión de los vectores de embedding y de las capas ocultas. |
| `ATTENTION_HEADS`| **8** | Número de "cabezas" en el mecanismo de Multi-Head Attention. |
| `NUMBER_ENCODERS`| **4** | Cantidad de capas apiladas en el bloque del Encoder. |
| `NUMBER_DECODERS`| **4** | Cantidad de capas apiladas en el bloque del Decoder. |
| `DROPOUT` | **0.1** | Tasa de abandono utilizada para la regularización. |

---

## Datasets de Entrenamiento
Para el entrenamiento del modelo se ha empleado el dataset **OPUS-100** para el par de idiomas inglés-español. 

Se ha realizado un filtrado de los datos para mejorar la calidad del corpus, y el análisis detallado de este proceso puede consultarse en el notebook `dataset.ipynb`.

---

## Tokenización y Vocabulario Custom (BPE)

En lugar de depender de tokenizadores preentrenados genéricos (como `cl100k_base` de OpenAI o el de GPT-2), este proyecto implementa **su propio tokenizador entrenado desde cero** sobre nuestro corpus bilingüe. Esto mejora significativamente el aprendizaje del algoritmo al estar adaptado específicamente al inglés y al español.

**Características del Tokenizador:**
* **Algoritmo:** Byte-Pair Encoding (BPE) a nivel de bytes (`ByteLevel`).
* **Tamaño del vocabulario:** 16.000 tokens (vocabulario compartido para ambos idiomas).
* **Tokens especiales:** * `<PAD>`: Para rellenar secuencias cortas.
    * `<START>`: Indica el inicio de la traducción.
    * `<END>`: Indica el final de la secuencia generada.
    * `<UNK>`: Para palabras fuera del vocabulario.

Contar con un vocabulario específico para el par inglés-español optimiza el espacio de embeddings. Esto no solo facilita el aprendizaje del Transformer, sino que aporta una generalización mucho más robusta ante textos no vistos.

---

## CLI

Se ha creado un flujo que permite interactuar con el modelo entrenado directamente a través de la consola.

### 1. Instalación de uv

Para gestionar el entorno y las dependencias de forma eficiente, es necesario instalar **[uv](https://github.com/astral-sh/uv)** . Utiliza el comando correspondiente a tu sistema operativo:

**macOS y Linux:**
```bash
# En macOS y Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
# En Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Configuración del proyecto

Una vez instalado `uv`, sigue estos pasos para configurar el repositorio:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/skaczylo/TranslatorEn2Es.git
   cd TranslatorEn2Es
   ```

2. **Sincronizar el entorno:**
   Este comando instalará automáticamente la versión de Python necesaria y todas las dependencias:
   ```bash
   uv sync
   ```
3. **Crear carpeta para el modelo. Es importante que la carpeta se llame model**
   ```bash
   mkdir model
   ```

4. **Descargar los pesos del modelo:**
   Descarga el archivo desde Google Drive directamente en la carpeta creada:
   ```bash
   uvx gdown "https://drive.google.com/file/d/1VSHrUPD1nuS1FAUT0cYzzy-twNzCBVtx/view?usp=drive_link" -O model/
   ```
   
### 3. Ejecución del traductor

Para iniciar la interfaz interactiva en la terminal y empezar a traducir, ejecuta:

```bash
uv run traductor
```




