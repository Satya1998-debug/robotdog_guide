import os

#paths
CHROMA_PATH = "./src/rag_server/chroma_db"
CSV_FILE_PATH = "./src/rag_server/ias_scraped_data/scraped_data.csv"
ROOMS_CSV_PATH = "./src/rag_server/ias_scraped_data/rooms.csv"

# Embedding & API Config
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# speech recognition model
SPEECH_RECOGNITION_MODEL = "vosk-model-en-us-0.22"
SPEECH_RECOGNITION_MODEL_PATH = "./src/rag_server/speech_model/"

# Memory choose between Simple or Summarized
MEMORY_TYPE = "Simple Memory" 
# Scraping Control
SCRAPE = {
    "need_scraping": False,
    "base_url": "https://www.ias.uni-stuttgart.de/",
    "data_dir": "./src/rag_server/ias_scraped_data",
    "max_pages": 500
}

# Voice Output Dir
OUTPUT_DIR = "./src/rag_server/output"

# Devices
MIC_DEVICE_INDEX = 1     # ReSpeaker Mic
SPEAKER_DEVICE_INDEX = 2   # local MacBook Pro Speaker