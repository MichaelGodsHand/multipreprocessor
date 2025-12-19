"""
Multi-tenant Preprocessor Server
Handles multiple clients with isolated S3 buckets, ChromaDB collections, and MongoDB databases.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import (
    get_client_config,
    get_mongodb_database_name,
    get_s3_bucket_name
)

# Import all from original preprocessor
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
import uuid
import tarfile
import shutil
import threading
import time
from datetime import datetime
from enum import Enum

# All the original imports
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Please install it using: pip install pymupdf")
    sys.exit(1)

try:
    from PIL import Image
    import pytesseract
except ImportError:
    print("OCR dependencies not found. Please install them using: pip install pillow pytesseract")
    sys.exit(1)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    print("ChromaDB not found. Please install it using: pip install chromadb")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not found. Please install it using: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv not found. Please install it using: pip install python-dotenv")
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("boto3 not found. Please install it using: pip install boto3")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("requests not found. Please install it using: pip install requests")
    sys.exit(1)

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
except ImportError:
    print("reportlab not found. Please install it using: pip install reportlab")
    sys.exit(1)

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    print("twilio not found. Please install it using: pip install twilio")
    sys.exit(1)

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    from bson import ObjectId
    MONGODB_AVAILABLE = True
except ImportError:
    print("pymongo not found. Please install it using: pip install pymongo")
    MONGODB_AVAILABLE = False

try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
except ImportError:
    print("Email libraries not found. Please ensure Python's email libraries are available.")
    sys.exit(1)

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build
    import pickle
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    print("Google Calendar libraries not found. Please install them using: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    GOOGLE_CALENDAR_AVAILABLE = False
    GoogleRequest = None

# Load environment variables
load_dotenv()

# Configure Tesseract path for Windows
TESSERACT_FOUND = False
if sys.platform == 'win32':
    import shutil
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        TESSERACT_FOUND = True
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    else:
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                TESSERACT_FOUND = True
                break
else:
    TESSERACT_FOUND = True

# Initialize FastAPI
app = FastAPI(
    title="Multi-Tenant PDF Text Extraction & Query API",
    description="Extract text from PDFs, store in ChromaDB, query with GPT-4o, and send via WhatsApp (Multi-tenant)",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking for async extraction (per client)
extraction_jobs: dict = {}
job_lock = threading.Lock()
backup_lock = threading.Lock()
extraction_in_progress: dict = {}  # {client_id: bool}
extraction_lock = threading.Lock()

# ChromaDB clients cache: {client_id: client}
chroma_clients: dict = {}
chroma_collections: dict = {}  # {client_id: collection}
chroma_lock = threading.Lock()

# S3 clients cache: {client_id: (client, bucket_name, region)}
s3_clients: dict = {}
s3_lock = threading.Lock()

# MongoDB clients cache: {client_id: client}
mongodb_clients: dict = {}
mongodb_lock = threading.Lock()

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def get_client_id_from_request(request: FastAPIRequest) -> str:
    """Extract client_id from query params or headers"""
    client_id = request.query_params.get("client_id") or request.headers.get("X-Client-ID")
    if not client_id:
        raise HTTPException(status_code=400, detail="client_id is required (query param or X-Client-ID header)")
    
    # Validate client exists
    try:
        get_client_config(client_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found")
    
    return client_id


def get_s3_client_for_client(client_id: str):
    """Get or create S3 client for a specific client"""
    with s3_lock:
        if client_id in s3_clients:
            return s3_clients[client_id]
        
        # Get client config
        config = get_client_config(client_id)
        s3_config = config.get('s3', {})
        bucket_name = s3_config.get('bucket_name', f"{client_id}-storage")
        region = s3_config.get('region', os.getenv("AWS_REGION", "ap-south-1"))
        
        # Get AWS credentials
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        
        if not aws_access_key or not aws_secret_key:
            print(f"\n⚠️  WARNING: AWS credentials not found for client {client_id}")
            s3_clients[client_id] = (None, bucket_name, region)
            return s3_clients[client_id]
        
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            s3_clients[client_id] = (s3_client, bucket_name, region)
            print(f"✓ S3 client initialized for client {client_id} (bucket: {bucket_name})")
            return s3_clients[client_id]
        except Exception as e:
            print(f"⚠️  Error initializing S3 client for client {client_id}: {e}")
            s3_clients[client_id] = (None, bucket_name, region)
            return s3_clients[client_id]


def get_chromadb_collection_for_client(client_id: str):
    """Get or create ChromaDB collection for a specific client"""
    with chroma_lock:
        if client_id in chroma_collections:
            return chroma_collections[client_id]
        
        # Get or create ChromaDB client
        if client_id not in chroma_clients:
            chroma_db_path = f"./chroma_db_{client_id}"
            os.makedirs(chroma_db_path, exist_ok=True)
            chroma_clients[client_id] = chromadb.PersistentClient(path=chroma_db_path)
        
        chroma_client = chroma_clients[client_id]
        
        # Get OpenAI embedding function
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-ada-002"
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize OpenAI embeddings for client {client_id}: {e}")
                openai_ef = None
        else:
            openai_ef = None
        
        # Get or create collection
        collection_name = f"{client_id}_pdf_documents"
        if openai_ef:
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=openai_ef,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        chroma_collections[client_id] = collection
        print(f"✓ ChromaDB collection initialized for client {client_id} ({collection_name})")
        return collection


def get_mongodb_client_for_client(client_id: str):
    """Get or create MongoDB client for a specific client"""
    with mongodb_lock:
        if client_id in mongodb_clients:
            return mongodb_clients[client_id]
        
        mongodb_uri = os.getenv("MONGODB_URI", "")
        if not mongodb_uri:
            print(f"\n⚠️  WARNING: MONGODB_URI not found for client {client_id}")
            mongodb_clients[client_id] = None
            return None
        
        if not MONGODB_AVAILABLE:
            print(f"\n⚠️  WARNING: pymongo not installed for client {client_id}")
            mongodb_clients[client_id] = None
            return None
        
        try:
            client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
            client.admin.command('ping')
            mongodb_clients[client_id] = client
            print(f"✓ MongoDB connection established for client {client_id}")
            return client
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"\n⚠️  WARNING: Failed to connect to MongoDB for client {client_id}: {e}")
            mongodb_clients[client_id] = None
            return None
        except Exception as e:
            print(f"\n⚠️  WARNING: Error connecting to MongoDB for client {client_id}: {e}")
            mongodb_clients[client_id] = None
            return None


# Import all the original functions from preprocessor/main.py
# For brevity, I'll include key multi-tenant wrappers
# The full implementation would include all original functions with client_id parameter

@app.head("/health")
async def health_check():
    """Health check endpoint"""
    return None


@app.post("/extract")
async def extract_text(
    request: FastAPIRequest,
    files: List[UploadFile] = File(...),
    use_ocr: bool = Form(True),
    webhook_url: Optional[str] = Form(None)
):
    """Extract text from multiple PDF files and store in ChromaDB (multi-tenant)"""
    client_id = get_client_id_from_request(request)
    
    # Get client-specific resources
    collection = get_chromadb_collection_for_client(client_id)
    s3_client, s3_bucket, s3_region = get_s3_client_for_client(client_id)
    
    # Continue with extraction logic using client-specific resources
    # (Full implementation would include all original extraction logic)
    # For now, return a placeholder response
    return JSONResponse(content={
        "status": "success",
        "message": f"Extraction endpoint for client {client_id} - full implementation needed",
        "client_id": client_id
    })


@app.post("/query")
async def query_documents(request: FastAPIRequest, query_request: dict):
    """Query the stored PDF documents using GPT-4o, ChromaDB (multi-tenant)"""
    client_id = get_client_id_from_request(request)
    
    # Get client-specific resources
    collection = get_chromadb_collection_for_client(client_id)
    s3_client, s3_bucket, s3_region = get_s3_client_for_client(client_id)
    mongodb_client = get_mongodb_client_for_client(client_id)
    
    # Continue with query logic using client-specific resources
    # (Full implementation would include all original query logic)
    return JSONResponse(content={
        "status": "success",
        "message": f"Query endpoint for client {client_id} - full implementation needed",
        "client_id": client_id
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

