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
import json
import tempfile
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

# Load environment variables
load_dotenv()

# Configure Tesseract path for Windows
TESSERACT_FOUND = False
if sys.platform == 'win32':
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

# Initialize shared services
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
twilio_client = TwilioClient(twilio_account_sid, twilio_auth_token) if twilio_account_sid and twilio_auth_token else None

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryRequest(BaseModel):
    query: str
    number: Optional[str] = None
    email: Optional[str] = None


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


def extract_text_with_ocr(page, page_num, pdf_name):
    """Extract text from a PDF page using OCR."""
    global TESSERACT_FOUND
    
    if not TESSERACT_FOUND:
        print(f"    ⚠ OCR skipped (Tesseract not available)")
        return ""
    
    try:
        print(f"    🔍 Running OCR on {pdf_name}&{page_num}...", end="", flush=True)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        ocr_text = pytesseract.image_to_string(img, lang='eng')
        print(" ✓ Done")
        return ocr_text.strip()
    except pytesseract.TesseractNotFoundError:
        TESSERACT_FOUND = False
        print(" ✗ Failed (Tesseract not found)")
        return ""
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return ""


def extract_text_from_pdf(pdf_bytes, pdf_filename, use_ocr=True):
    """Extract text from PDF file page by page."""
    pdf_text = {}
    page_objects = {}  # Store page objects for S3 upload
    
    try:
        pdf_name = Path(pdf_filename).stem
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        print(f"\n{'='*80}")
        print(f"📄 Processing PDF: {pdf_filename}")
        print(f"📊 Total pages: {len(doc)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print(f"{'='*80}\n")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_identifier = f"{pdf_name}&{page_num + 1}"
            
            print(f"  📖 Page {page_num + 1}/{len(doc)} ({page_identifier})")
            
            print(f"    📝 Extracting regular text...", end="", flush=True)
            regular_text = page.get_text()
            regular_char_count = len(regular_text.strip())
            print(f" ✓ ({regular_char_count} chars)")
            
            ocr_text = ""
            ocr_char_count = 0
            if use_ocr:
                ocr_text = extract_text_with_ocr(page, page_num + 1, pdf_name)
                ocr_char_count = len(ocr_text.strip())
                if ocr_text:
                    print(f"      ✓ OCR extracted {ocr_char_count} chars")
            
            if regular_text.strip() and ocr_text.strip():
                if regular_text.strip() in ocr_text or len(regular_text.strip()) < 50:
                    combined_text = ocr_text
                    print(f"    💡 Using OCR text only (regular text contained in OCR)")
                else:
                    combined_text = f"{regular_text}\n\n[OCR Text from Images:]\n{ocr_text}"
                    print(f"    💡 Combined regular + OCR text")
            elif ocr_text.strip():
                combined_text = f"[OCR Text:]\n{ocr_text}"
                print(f"    💡 Using OCR text only")
            else:
                combined_text = regular_text
                print(f"    💡 Using regular text only")
            
            pdf_text[page_identifier] = combined_text
            page_objects[page_identifier] = page
            
            total_chars = len(combined_text.strip())
            print(f"    ✅ Page {page_identifier} complete - Total: {total_chars} chars")
            print(f"    {'-'*76}")
        
        print(f"\n{'='*80}")
        print(f"✅ PDF Processing Complete: {pdf_filename}")
        print(f"📊 Total pages processed: {len(pdf_text)}")
        print(f"{'='*80}\n")
        
        return pdf_text, page_objects, doc
        
    except Exception as e:
        print(f"\n❌ Error extracting text from {pdf_filename}: {str(e)}\n")
        raise Exception(f"Error extracting text from {pdf_filename}: {str(e)}")


def store_in_chromadb(client_id: str, collection, page_identifier: str, text: str, pdf_name: str, page_number: int):
    """Store extracted text in ChromaDB with metadata. Prevents duplicates."""
    try:
        print(f"    💾 Storing {page_identifier} in ChromaDB...", end="", flush=True)
        
        # Check if this page already exists (prevent duplicates)
        existing = collection.get(
            where={"page_identifier": page_identifier},
            limit=1
        )
        
        if existing['ids']:
            # Page exists, update it instead of creating duplicate
            print(f" (updating existing)...", end="", flush=True)
            collection.update(
                ids=[existing['ids'][0]],
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }]
            )
            print(" ✓ Updated")
        else:
            # New page, add it
            doc_id = f"{page_identifier}"
            collection.add(
                documents=[text],
                metadatas=[{
                    "page_identifier": page_identifier,
                    "pdf_name": pdf_name,
                    "page_number": str(page_number),
                    "char_count": str(len(text))
                }],
                ids=[doc_id]
            )
            print(" ✓ Stored")
        
        return True
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return False


def upload_page_to_s3(client_id: str, s3_client, bucket_name: str, page, page_identifier: str):
    """Convert PDF page to image and upload to S3."""
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping upload for {page_identifier}")
        return None
    
    try:
        print(f"    📤 Uploading {page_identifier} to S3...", end="", flush=True)
        
        # Render page as high-quality image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        
        # S3 key (filename in bucket)
        s3_key = f"{page_identifier}.png"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=img_bytes,
            ContentType='image/png'
        )
        
        # Generate S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def upload_full_pdf_to_s3(client_id: str, s3_client, bucket_name: str, pdf_bytes: bytes, pdf_filename: str):
    """Upload the entire PDF file to S3."""
    if s3_client is None:
        print(f"    ⚠ S3 client not configured, skipping full PDF upload for {pdf_filename}")
        return None
    
    try:
        print(f"    📤 Uploading full PDF {pdf_filename} to S3...", end="", flush=True)
        
        # Sanitize filename for S3 key (remove special characters, keep extension)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in pdf_filename if c.isalnum() or c in ('-', '_', '.'))
        if not safe_filename.endswith('.pdf'):
            safe_filename = f"{safe_filename}.pdf"
        
        # S3 key (store in pdfs/ subfolder for organization)
        s3_key = f"pdfs/{timestamp}_{safe_filename}"
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'original_filename': pdf_filename,
                'uploaded_at': timestamp
            }
        )
        
        # Generate S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        print(f" ✓ Uploaded")
        return s3_url
        
    except ClientError as e:
        print(f" ✗ Failed (S3 Error: {e.response['Error']['Message']})")
        return None
    except Exception as e:
        print(f" ✗ Failed ({str(e)})")
        return None


def create_compiled_pdf_from_images(client_id: str, s3_client, bucket_name: str, region: str, s3_image_urls: List[str], user_number: str, query: str):
    """Download images from S3, compile them into a single PDF, and upload to S3."""
    if s3_client is None:
        print(f"  ⚠ S3 client not configured, cannot create compiled PDF")
        return None
    
    try:
        print(f"  📄 Creating compiled PDF from {len(s3_image_urls)} images...")
        
        if not s3_image_urls:
            return None
        
        # Create a temporary PDF file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()
        
        # Create PDF with ReportLab
        c = canvas.Canvas(temp_pdf_path, pagesize=A4)
        page_width, page_height = A4
        
        # Download and add each image to PDF
        for idx, img_url in enumerate(s3_image_urls, 1):
            print(f"    📥 Processing image {idx}/{len(s3_image_urls)}...", end="", flush=True)
            
            try:
                # Download image from S3
                response = requests.get(img_url, timeout=30)
                response.raise_for_status()
                
                # Open image with PIL
                img = Image.open(BytesIO(response.content))
                
                # Calculate dimensions to fit page while maintaining aspect ratio
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Fit to page with margins
                margin = 20
                available_width = page_width - (2 * margin)
                available_height = page_height - (2 * margin)
                
                if available_width * aspect <= available_height:
                    display_width = available_width
                    display_height = available_width * aspect
                else:
                    display_height = available_height
                    display_width = available_height / aspect
                
                # Center image on page
                x = (page_width - display_width) / 2
                y = (page_height - display_height) / 2
                
                # Draw image
                img_reader = ImageReader(BytesIO(response.content))
                c.drawImage(img_reader, x, y, width=display_width, height=display_height)
                
                # Add new page if not last image
                if idx < len(s3_image_urls):
                    c.showPage()
                
                print(" ✓")
                
            except Exception as e:
                print(f" ✗ Failed: {str(e)}")
                continue
        
        # Save PDF
        c.save()
        print(f"  ✓ PDF created successfully")
        
        # Upload to S3
        print(f"  📤 Uploading compiled PDF to S3...", end="", flush=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"query_{user_number}_{timestamp}.pdf"
        
        # Read PDF file
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket_name,
            Key=f"compiled_pdfs/{pdf_filename}",
            Body=pdf_bytes,
            ContentType='application/pdf',
            Metadata={
                'user_number': user_number,
                'query': query[:200],
                'page_count': str(len(s3_image_urls))
            }
        )
        
        # Generate S3 URL
        compiled_pdf_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/compiled_pdfs/{pdf_filename}"
        print(f" ✓ Uploaded")
        print(f"  ✓ Compiled PDF URL: {compiled_pdf_url}")
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        return compiled_pdf_url
        
    except Exception as e:
        print(f"  ✗ Failed to create compiled PDF: {str(e)}")
        return None


def send_whatsapp_message(to_number: str, summary: str, pdf_url: str = None):
    """Send WhatsApp message with summary and optional PDF attachment."""
    if twilio_client is None:
        print(f"  ⚠ Twilio client not configured, skipping WhatsApp message")
        return {"status": "skipped", "reason": "Twilio not configured"}
    
    try:
        print(f"\n  📱 Sending WhatsApp message...")
        print(f"    → To: {to_number}")
        print(f"    → Summary length: {len(summary)} chars")
        print(f"    → PDF URL: {pdf_url if pdf_url else 'None'}")
        
        # Format phone number for WhatsApp
        formatted_number = f"whatsapp:{to_number}" if not to_number.startswith('whatsapp:') else to_number
        
        # Prepare message body
        message_body = summary
        
        # Create message parameters
        message_params = {
            'from_': twilio_whatsapp_number,
            'body': message_body,
            'to': formatted_number
        }
        
        # Add PDF if available
        if pdf_url:
            try:
                test_response = requests.head(pdf_url, timeout=5)
                if test_response.status_code == 200:
                    message_params['media_url'] = [pdf_url]
            except Exception:
                message_params['media_url'] = [pdf_url]  # Try anyway
        
        # Send message
        message = twilio_client.messages.create(**message_params)
        
        print(f"  ✓ WhatsApp message sent successfully!")
        print(f"    → Message SID: {message.sid}")
        
        return {
            "status": "success",
            "message_sid": message.sid,
            "twilio_status": message.status,
            "pdf_url_sent": pdf_url
        }
        
    except Exception as e:
        print(f"  ✗ Failed to send WhatsApp message: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


def send_email_message(to_email: str, summary: str, pdf_url: str = None, query: str = None):
    """Send email with summary and optional PDF attachment."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from_email = os.getenv("SMTP_FROM_EMAIL", smtp_username)
    
    if not smtp_username or not smtp_password:
        print(f"  ⚠ SMTP credentials not configured, skipping email")
        return {"status": "skipped", "reason": "SMTP not configured"}
    
    try:
        print(f"\n  📧 Sending email...")
        print(f"    → To: {to_email}")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_from_email
        msg['To'] = to_email
        msg['Subject'] = f"Query Response: {query[:50]}" if query else "Query Response"
        
        # Add body to email
        body = f"""
Hello,

Thank you for your query. Please find the detailed information below:

{summary}

"""
        
        if pdf_url:
            body += f"\nA detailed PDF document has been attached for your reference.\n"
            body += f"\nYou can also access it directly here: {pdf_url}\n"
        
        body += """
Best regards,
Support Team
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF if URL is provided
        if pdf_url:
            try:
                pdf_response = requests.get(pdf_url, timeout=30)
                pdf_response.raise_for_status()
                
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(pdf_response.content)
                encoders.encode_base64(part)
                
                filename = pdf_url.split('/')[-1] if '/' in pdf_url else "query_response.pdf"
                if not filename.endswith('.pdf'):
                    filename = "query_response.pdf"
                
                part.add_header('Content-Disposition', f'attachment; filename= {filename}')
                msg.attach(part)
            except Exception:
                pass
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_from_email, to_email, text)
        server.quit()
        
        print(f"  ✓ Email sent successfully!")
        
        return {
            "status": "success",
            "to": to_email,
            "subject": msg['Subject'],
            "pdf_attached": pdf_url is not None
        }
        
    except Exception as e:
        print(f"  ✗ Failed to send email: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


def send_messages_background(client_id: str, user_number: Optional[str], user_email: Optional[str], summary: str, compiled_pdf_url: Optional[str], query: str):
    """Background function to send WhatsApp and Email messages."""
    try:
        print(f"\n  🔄 Background: Starting message sending...")
        
        # Send WhatsApp message
        whatsapp_status = {"status": "skipped", "reason": "No phone number provided"}
        if user_number:
            whatsapp_status = send_whatsapp_message(user_number, summary, compiled_pdf_url)
        
        # Send Email message
        email_status = {"status": "skipped", "reason": "No email provided"}
        if user_email:
            email_status = send_email_message(user_email, summary, compiled_pdf_url, query)
        
        print(f"  ✅ Background: Message sending completed")
        print(f"    📱 WhatsApp: {whatsapp_status['status']}")
        print(f"    📧 Email: {email_status['status']}")
        
    except Exception as e:
        print(f"  ❌ Background: Error sending messages: {str(e)}")


def send_webhook_notification(webhook_url: str, job_id: str, job_data: dict, max_retries: int = 3):
    """Send webhook notification with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"  📡 Sending webhook notification (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                webhook_url,
                json=job_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            print(f"  ✓ Webhook notification sent successfully")
            return True
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ Webhook notification failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  ⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Webhook notification failed after {max_retries} attempts")
                return False
    return False


def process_extraction_job(client_id: str, job_id: str, files_data: list, use_ocr: bool, webhook_url: Optional[str] = None):
    """Background function to process PDF extraction."""
    # Set extraction flag
    with extraction_lock:
        extraction_in_progress[client_id] = True
    
    try:
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.PROCESSING
            extraction_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        print("\n" + "="*80)
        print(f"🚀 Starting batch PDF extraction (Job ID: {job_id}, Client: {client_id})")
        print(f"📁 Total files: {len(files_data)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print("="*80)
        
        # Get client-specific resources
        collection = get_chromadb_collection_for_client(client_id)
        s3_client, bucket_name, region = get_s3_client_for_client(client_id)
        
        results = {}
        errors = []
        stored_count = 0
        s3_upload_count = 0
        pdf_s3_urls = {}
        
        for idx, (filename, contents) in enumerate(files_data, 1):
            print(f"\n📦 Processing file {idx}/{len(files_data)}: {filename}")
            
            if not filename.lower().endswith('.pdf'):
                error_msg = f"{filename}: Not a PDF file"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
            
            try:
                print(f"  📥 File size: {len(contents)} bytes")
                
                # Upload full PDF to S3 first
                full_pdf_s3_url = upload_full_pdf_to_s3(client_id, s3_client, bucket_name, contents, filename)
                if full_pdf_s3_url:
                    pdf_s3_urls[filename] = full_pdf_s3_url
                
                # Extract text
                extracted_text, page_objects, doc = extract_text_from_pdf(contents, filename, use_ocr)
                
                # Store each page in ChromaDB and upload to S3
                pdf_name = Path(filename).stem
                print(f"\n  💾 Storing pages in ChromaDB and uploading to S3...")
                
                for page_identifier, text in extracted_text.items():
                    page_number = int(page_identifier.split('&')[1])
                    
                    # Store in ChromaDB
                    if store_in_chromadb(client_id, collection, page_identifier, text, pdf_name, page_number):
                        stored_count += 1
                    
                    # Upload page image to S3
                    page = page_objects[page_identifier]
                    s3_url = upload_page_to_s3(client_id, s3_client, bucket_name, page, page_identifier)
                    if s3_url:
                        s3_upload_count += 1
                
                # Close the document
                doc.close()
                
                results.update(extracted_text)
                print(f"  ✅ Successfully extracted and stored {len(extracted_text)} pages from {filename}")
                
            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                errors.append(error_msg)
                print(f"  ❌ Failed: {error_msg}")
        
        # Prepare response data
        response_data = {
            "job_id": job_id,
            "status": "success" if results else "failed",
            "total_files_processed": len(files_data),
            "total_pages_extracted": len(results),
            "total_pages_stored_in_db": stored_count,
            "total_pages_uploaded_to_s3": s3_upload_count,
            "ocr_enabled": use_ocr,
            "ocr_available": TESSERACT_FOUND,
            "page_identifiers": list(results.keys()),
            "pdf_s3_urls": pdf_s3_urls
        }
        
        if errors:
            response_data["errors"] = errors
        
        # Update job status
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.COMPLETED if results else JobStatus.FAILED
            extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            extraction_jobs[job_id]["result"] = response_data
        
        print("\n" + "="*80)
        print(f"🎉 Batch extraction complete! (Job ID: {job_id})")
        print(f"✅ Success: {len(results)} pages extracted")
        print(f"💾 Stored: {stored_count} pages in ChromaDB")
        print(f"📤 Uploaded: {s3_upload_count} pages to S3")
        if errors:
            print(f"⚠ Errors: {len(errors)} files failed")
        print("="*80 + "\n")
        
        # Send webhook notification if provided
        if webhook_url:
            send_webhook_notification(webhook_url, job_id, response_data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Job {job_id} failed: {error_msg}\n")
        
        with job_lock:
            extraction_jobs[job_id]["status"] = JobStatus.FAILED
            extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
            extraction_jobs[job_id]["error"] = error_msg
            extraction_jobs[job_id]["result"] = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
        
        if webhook_url:
            failure_data = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg
            }
            send_webhook_notification(webhook_url, job_id, failure_data)
    finally:
        with extraction_lock:
            extraction_in_progress[client_id] = False


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
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Get client-specific resources
    collection = get_chromadb_collection_for_client(client_id)
    s3_client, bucket_name, region = get_s3_client_for_client(client_id)
    
    # If webhook_url is provided, process asynchronously
    if webhook_url:
        job_id = str(uuid.uuid4())
        
        # Read all file contents into memory
        files_data = []
        for file in files:
            contents = await file.read()
            files_data.append((file.filename, contents))
        
        # Initialize job tracking
        with job_lock:
            extraction_jobs[job_id] = {
                "job_id": job_id,
                "client_id": client_id,
                "status": JobStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "total_files": len(files),
                "use_ocr": use_ocr,
                "webhook_url": webhook_url,
                "result": None,
                "error": None
            }
        
        # Start background processing
        print(f"\n🔄 Starting async extraction job: {job_id} (client: {client_id})")
        threading.Thread(
            target=process_extraction_job,
            args=(client_id, job_id, files_data, use_ocr, webhook_url),
            daemon=True
        ).start()
        
        return JSONResponse(content={
            "status": "processing",
            "job_id": job_id,
            "client_id": client_id,
            "message": "Extraction started. You will be notified via webhook when complete.",
            "webhook_url": webhook_url,
            "check_status_url": f"/extract/status/{job_id}"
        })
    
    # Synchronous processing
    with extraction_lock:
        extraction_in_progress[client_id] = True
    
    try:
        print("\n" + "="*80)
        print(f"🚀 Starting batch PDF extraction (synchronous, client: {client_id})")
        print(f"📁 Total files received: {len(files)}")
        print(f"🔧 OCR enabled: {use_ocr}")
        print("="*80)
        
        results = {}
        errors = []
        stored_count = 0
        s3_upload_count = 0
        pdf_s3_urls = {}
        
        for idx, file in enumerate(files, 1):
            print(f"\n📦 Processing file {idx}/{len(files)}: {file.filename}")
            
            if not file.filename.lower().endswith('.pdf'):
                error_msg = f"{file.filename}: Not a PDF file"
                errors.append(error_msg)
                print(f"  ❌ {error_msg}")
                continue
            
            try:
                print(f"  📥 Reading file contents...")
                contents = await file.read()
                print(f"  ✓ File size: {len(contents)} bytes")
        
                # Upload full PDF to S3 first
                full_pdf_s3_url = upload_full_pdf_to_s3(client_id, s3_client, bucket_name, contents, file.filename)
                if full_pdf_s3_url:
                    pdf_s3_urls[file.filename] = full_pdf_s3_url
        
                # Extract text
                extracted_text, page_objects, doc = extract_text_from_pdf(contents, file.filename, use_ocr)
                
                # Store each page in ChromaDB and upload to S3
                pdf_name = Path(file.filename).stem
                print(f"\n  💾 Storing pages in ChromaDB and uploading to S3...")
                
                for page_identifier, text in extracted_text.items():
                    page_number = int(page_identifier.split('&')[1])
                    
                    # Store in ChromaDB
                    if store_in_chromadb(client_id, collection, page_identifier, text, pdf_name, page_number):
                        stored_count += 1
                    
                    # Upload page image to S3
                    page = page_objects[page_identifier]
                    s3_url = upload_page_to_s3(client_id, s3_client, bucket_name, page, page_identifier)
                    if s3_url:
                        s3_upload_count += 1
                
                # Close the document
                doc.close()
                
                results.update(extracted_text)
                print(f"  ✅ Successfully extracted and stored {len(extracted_text)} pages from {file.filename}")
                
            except Exception as e:
                error_msg = f"{file.filename}: {str(e)}"
                errors.append(error_msg)
                print(f"  ❌ Failed: {error_msg}")
        
        response = {
            "status": "success" if results else "failed",
            "client_id": client_id,
            "total_files_processed": len(files),
            "total_pages_extracted": len(results),
            "total_pages_stored_in_db": stored_count,
            "total_pages_uploaded_to_s3": s3_upload_count,
            "ocr_enabled": use_ocr,
            "ocr_available": TESSERACT_FOUND,
            "page_identifiers": list(results.keys()),
            "pdf_s3_urls": pdf_s3_urls
        }
        
        if errors:
            response["errors"] = errors
        
        print("\n" + "="*80)
        print(f"🎉 Batch extraction complete!")
        print(f"✅ Success: {len(results)} pages extracted")
        print(f"💾 Stored: {stored_count} pages in ChromaDB")
        print(f"📤 Uploaded: {s3_upload_count} pages to S3")
        if errors:
            print(f"⚠ Errors: {len(errors)} files failed")
        print("="*80 + "\n")
        
        return JSONResponse(content=response)
    finally:
        with extraction_lock:
            extraction_in_progress[client_id] = False


@app.get("/extract/status/{job_id}")
async def get_extraction_status(job_id: str):
    """Get the status of an extraction job."""
    with job_lock:
        if job_id not in extraction_jobs:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        job = extraction_jobs[job_id]
        
        response = {
            "job_id": job_id,
            "client_id": job.get("client_id"),
            "status": job["status"].value if isinstance(job["status"], JobStatus) else job["status"],
            "created_at": job["created_at"],
            "total_files": job["total_files"],
            "use_ocr": job["use_ocr"],
            "webhook_url": job.get("webhook_url")
        }
        
        if "started_at" in job:
            response["started_at"] = job["started_at"]
        
        if "completed_at" in job:
            response["completed_at"] = job["completed_at"]
        
        if job["status"] in [JobStatus.COMPLETED, JobStatus.FAILED] or (isinstance(job["status"], str) and job["status"] in ["completed", "failed"]):
            response["result"] = job["result"]
            if job.get("error"):
                response["error"] = job["error"]
        
        return JSONResponse(content=response)


@app.post("/query")
async def query_documents(request: FastAPIRequest, query_request: QueryRequest):
    """Query the stored PDF documents using GPT-4o, ChromaDB (multi-tenant)"""
    client_id = get_client_id_from_request(request)
    
    query = query_request.query
    user_number = query_request.number
    user_email = query_request.email
    
    # Validate that at least one contact method is provided
    if not user_number and not user_email:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'number' or 'email' must be provided"
        )
    
    print("\n" + "="*80)
    print(f"🔍 Processing query (client: {client_id}): {query}")
    print(f"📱 User number: {user_number if user_number else 'Not provided'}")
    print(f"📧 User email: {user_email if user_email else 'Not provided'}")
    print("="*80)
    
    try:
        # Get client-specific resources
        collection = get_chromadb_collection_for_client(client_id)
        s3_client, bucket_name, region = get_s3_client_for_client(client_id)
        
        # Query ChromaDB for relevant documents
        print(f"  📊 Searching ChromaDB for relevant documents...")
        
        query_params = {
            'query_texts': [query],
            'n_results': 20
        }
        
        results = collection.query(**query_params)
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            print(f"  ⚠ No documents found in ChromaDB")
            return JSONResponse(content={
                "status": "no_results",
                "summary": "No relevant documents found in the database.",
                "pages": [],
                "s3_images": [],
                "compiled_pdf_url": None,
                "whatsapp_status": {"status": "skipped", "reason": "No results"},
                "email_status": {"status": "skipped", "reason": "No results"}
            })
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Deduplicate by page_identifier
        seen_pages = {}
        for doc, meta, dist in zip(documents, metadatas, distances):
            page_id = meta['page_identifier']
            relevance = 1 - dist
            
            if page_id not in seen_pages or relevance > seen_pages[page_id]['relevance']:
                seen_pages[page_id] = {
                    'doc': doc,
                    'meta': meta,
                    'dist': dist,
                    'relevance': relevance
                }
        
        # Sort by relevance and take top results
        sorted_pages = sorted(seen_pages.items(), key=lambda x: x[1]['relevance'], reverse=True)
        
        deduplicated_docs = []
        deduplicated_metas = []
        deduplicated_distances = []
        
        for page_id, page_data in sorted_pages[:15]:
            deduplicated_docs.append(page_data['doc'])
            deduplicated_metas.append(page_data['meta'])
            deduplicated_distances.append(page_data['dist'])
        
        print(f"  ✓ Found {len(deduplicated_docs)} unique relevant documents")
        
        # Prepare context for GPT-4o
        context_parts = []
        pages_used = []
        
        for doc, metadata, distance in zip(deduplicated_docs, deduplicated_metas, deduplicated_distances):
            page_identifier = metadata['page_identifier']
            pages_used.append(page_identifier)
            context_parts.append(f"[Source: {page_identifier}]\n{doc}\n")
        
        context = "\n".join(context_parts)
        
        # Check if OpenAI client is available
        if openai_client is None:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file."
            )
        
        # Query GPT-4o
        print(f"\n  🤖 Querying GPT-4o...")
        
        system_prompt = """You are a helpful assistant that answers questions based on PDF document content.
You MUST use the provided document excerpts to answer the user's query. 
Your response MUST be in JSON format with exactly two fields:
1. "summary": A comprehensive answer based ONLY on the provided document excerpts
2. "pages_used": An array of page identifiers that you used from the provided excerpts"""
        
        user_prompt = f"""Query: "{query}"

Below are document excerpts. Answer the query using ONLY information from these excerpts.

Document excerpts:
{context}

Respond in JSON format:
{{
  "summary": "Your detailed answer based on the excerpts above.",
  "pages_used": ["page-identifier-1", "page-identifier-2", ...]
}}"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        summary = result.get("summary", "")
        pages_actually_used = result.get("pages_used", [])
        
        # Ensure pages_actually_used are valid
        pages_actually_used = [p for p in pages_actually_used if p in pages_used]
        
        if not pages_actually_used:
            return JSONResponse(content={
                "status": "no_results",
                "summary": "No relevant documents found in the database for this query.",
                "pages": [],
                "s3_images": [],
                "compiled_pdf_url": None,
                "whatsapp_status": {"status": "skipped", "reason": "No results"},
                "email_status": {"status": "skipped", "reason": "No results"}
            })
        
        print(f"  ✓ GPT-4o response generated")
        print(f"  📄 GPT-4o used {len(pages_actually_used)} out of {len(pages_used)} retrieved pages")
        
        # Fetch S3 URLs for the pages used
        s3_images = []
        if s3_client is not None:
            print(f"\n  🔗 Fetching S3 image URLs...")
            for page_identifier in pages_actually_used:
                s3_key = f"{page_identifier}.png"
                s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
                
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                    s3_images.append(s3_url)
                    print(f"    ✓ Found: {s3_url}")
                except ClientError:
                    print(f"    ⚠ Not found in S3: {page_identifier}")
        
        # Create compiled PDF from images
        compiled_pdf_url = None
        pdf_identifier = user_number if user_number else (user_email.split('@')[0] if user_email else "unknown")
        if s3_client is not None and s3_images:
            print(f"\n  📚 Creating compiled PDF...")
            compiled_pdf_url = create_compiled_pdf_from_images(client_id, s3_client, bucket_name, region, s3_images, pdf_identifier, query)
        
        # Start background thread for WhatsApp and Email
        if user_number or user_email:
            print(f"\n  🔄 Starting background thread for message sending...")
            threading.Thread(
                target=send_messages_background,
                args=(client_id, user_number, user_email, summary, compiled_pdf_url, query),
                daemon=True
            ).start()
        
        print(f"\n{'='*80}")
        print(f"✅ Query completed successfully")
        print(f"📄 Pages retrieved: {len(pages_used)}, Pages used: {len(pages_actually_used)}")
        print(f"🖼️  S3 images found: {len(s3_images)}")
        print(f"📚 Compiled PDF: {'Created' if compiled_pdf_url else 'Failed'}")
        print(f"{'='*80}\n")
        
        # Return response immediately
        return JSONResponse(content={
            "status": "success",
            "summary": summary,
            "pages": pages_actually_used,
            "s3_images": s3_images,
            "compiled_pdf_url": compiled_pdf_url,
            "whatsapp_status": {"status": "processing", "message": "WhatsApp message is being sent in the background"},
            "email_status": {"status": "processing", "message": "Email is being sent in the background"}
        })
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
