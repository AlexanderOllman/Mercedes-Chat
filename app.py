from flask import Flask, render_template, jsonify, request
import subprocess
import os
import sys
import logging
import json
import boto3
import time
from io import BytesIO
import base64
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import weaviate
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Disable template caching to ensure templates are always reloaded
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a specific logger for Weaviate connections with more verbose output
weaviate_logger = logging.getLogger('weaviate_connection')
weaviate_logger.setLevel(logging.DEBUG)  # Set to DEBUG for more detailed logs

# Add a file handler for Weaviate logs
weaviate_file_handler = logging.FileHandler('weaviate_connection.log')
weaviate_file_handler.setLevel(logging.DEBUG)
weaviate_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
weaviate_logger.addHandler(weaviate_file_handler)

# Embedding model setup
def load_embedding_model():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

# Initialize the embedding model
embedding_model = load_embedding_model()

# Weaviate client setup
def connect_to_weaviate(weaviate_url=None, weaviate_token=None, verbose=False):
    """Connect to Weaviate instance using provided credentials or environment variables
    
    Args:
        weaviate_url (str, optional): URL of the Weaviate instance
        weaviate_token (str, optional): Auth token for Weaviate
        verbose (bool, optional): Whether to enable verbose logging
    
    Returns:
        weaviate.Client or None: Connected client or None if connection failed
    """
    # Set local logger to DEBUG if verbose mode is enabled
    if verbose:
        weaviate_logger.setLevel(logging.DEBUG)
        # Also add a stream handler to show debug logs in console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        weaviate_logger.addHandler(console_handler)
    
    url = weaviate_url or os.getenv('WEAVIATE_URL', 'weaviate.poc-weaviate.svc.cluster.local')
    token = weaviate_token or os.getenv('WEAVIATE_TOKEN')
    
    weaviate_logger.debug(f"Attempting to connect to Weaviate at: {url}")
    weaviate_logger.debug(f"Using auth token: {'Yes' if token else 'No'}")
    
    headers = {}
    if token:
        headers = {"x-auth-token": token}
        weaviate_logger.debug("Auth token provided, adding to request headers")
    
    try:
        weaviate_logger.debug(f"Connection parameters: HTTP Host={url}, HTTP Port=80, GRPC Host=weaviate-grpc.poc-weaviate.svc.cluster.local, GRPC Port=50051")
        
        client = weaviate.connect_to_custom(
            http_host=url,
            http_port=80,
            http_secure=False,
            grpc_host="weaviate-grpc.poc-weaviate.svc.cluster.local",
            grpc_port=50051,
            grpc_secure=False,
            headers=headers,
            skip_init_checks=False
        )
        
        weaviate_logger.debug("Weaviate client created, checking if ready...")
        
        if client.is_ready():
            weaviate_logger.info("Successfully connected to Weaviate")
            logger.info("Successfully connected to Weaviate")
            
            # Get and log cluster status information
            try:
                meta = client.get_meta()
                weaviate_logger.debug(f"Weaviate version: {meta.get('version', 'unknown')}")
                weaviate_logger.debug(f"Weaviate status: {meta}")
            except Exception as meta_e:
                weaviate_logger.warning(f"Connected but couldn't retrieve metadata: {str(meta_e)}")
            
            return client
        else:
            weaviate_logger.error("Failed to connect to Weaviate: is_ready() returned False")
            logger.error("Failed to connect to Weaviate")
            return None
    except weaviate.exceptions.WeaviateConnectionError as conn_err:
        weaviate_logger.error(f"Weaviate connection error: {str(conn_err)}")
        logger.error(f"Error connecting to Weaviate: {str(conn_err)}")
        return None
    except weaviate.exceptions.WeaviateAuthenticationError as auth_err:
        weaviate_logger.error(f"Weaviate authentication error: {str(auth_err)}")
        logger.error(f"Weaviate authentication error: {str(auth_err)}")
        return None
    except Exception as e:
        weaviate_logger.error(f"Unexpected error connecting to Weaviate: {str(e)}")
        weaviate_logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Error connecting to Weaviate: {str(e)}")
        return None

# Initialize Weaviate collection
def initialize_weaviate_collection(client, collection_name="MercedesImageEmbedding"):
    """Initialize or get the Weaviate collection"""
    import weaviate.classes.config as wc
    from weaviate.classes.config import Configure, DataType, Property
    
    try:
        # Check if collection exists
        if client.collections.exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return client.collections.get(collection_name)
        
        # Create collection
        client.collections.create(
            name=collection_name,
            properties=[
                Property(name="s3_key", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            vectorizer_config=wc.Configure.Vectorizer.none(),
        )
        
        logger.info(f"Created collection {collection_name}")
        return client.collections.get(collection_name)
    except Exception as e:
        logger.error(f"Error initializing Weaviate collection: {str(e)}")
        return None

# S3 client function
def get_s3_client(endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None):
    """Get an S3 client with the provided credentials"""
    # Use environment variables as defaults
    endpoint = endpoint_url or os.getenv('S3_ENDPOINT_URL')
    access_key = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # Create the S3 client
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    return s3_client

# Image processing functions
def read_image_from_s3(s3_client, bucket_name, object_key):
    """Read image from S3 bucket"""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        image_content = response['Body'].read()
        image = Image.open(BytesIO(image_content))
        return image
    except Exception as e:
        logger.error(f"Error reading image from S3: {str(e)}")
        return None

def image_to_numpy_array(image, target_size=(224, 224)):
    """Convert image to numpy array for embedding"""
    try:
        image = image.resize(target_size)
        image = image.convert('RGB')  # Ensure image is in RGB format
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        logger.error(f"Error converting image to numpy array: {str(e)}")
        return None

def get_image_embedding(model, img_array):
    """Generate embedding from image array"""
    try:
        embedding = model.predict(img_array)
        return embedding.flatten()
    except Exception as e:
        logger.error(f"Error generating image embedding: {str(e)}")
        return None

def store_embedding_in_weaviate(collection, s3_key, text, embedding):
    """Store embedding in Weaviate collection"""
    try:
        collection.data.insert(
            properties={
                "s3_key": s3_key,
                "text": text
            },
            vector=embedding.tolist()
        )
        return True
    except Exception as e:
        logger.error(f"Error storing embedding in Weaviate: {str(e)}")
        return False

# Store only the text embedding (we'll add this for text embedding)
def get_text_embedding(text):
    """Generate a basic text embedding using random values for demonstration purposes.
    In a production app, you would use a proper text embedding model."""
    # For this demo, we're using random embeddings
    # In a real app, use a text embedding model
    logger.info(f"Generating text embedding for: {text[:50]}...")
    return np.random.random(128).astype('float32')  # Small dimension for demo purposes

# Store embedding in Weaviate collection without using image
def store_text_embedding_in_weaviate(collection, s3_key, text):
    """Store text embedding in Weaviate collection"""
    try:
        logger.info(f"Generating embedding for s3_key: {s3_key}")
        # Generate a text embedding
        embedding = get_text_embedding(text)
        
        logger.info(f"Storing embedding in Weaviate collection: {collection.name}")
        collection.data.insert(
            properties={
                "s3_key": s3_key,
                "text": text
            },
            vector=embedding.tolist()
        )
        logger.info(f"Successfully stored embedding for {s3_key}")
        return True
    except Exception as e:
        logger.error(f"Error storing text embedding in Weaviate: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        query = request.json.get('query', '')
        logger.info(f"Received search query: {query}")
        
        # Get S3 and Weaviate credentials from request
        s3_endpoint = request.json.get('s3_endpoint')
        aws_access_key_id = request.json.get('aws_access_key_id')
        aws_secret_access_key = request.json.get('aws_secret_access_key')
        bucket_name = request.json.get('s3_bucket_name') or os.getenv('S3_BUCKET_NAME', 'default-bucket')
        weaviate_url = request.json.get('weaviate_url')
        weaviate_token = request.json.get('weaviate_token')
        
        logger.info(f"Connecting to Weaviate at: {weaviate_url or 'default URL'}")
        # Connect to Weaviate
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token)
        if not weaviate_client:
            logger.error("Failed to connect to Weaviate")
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Weaviate',
                'logs': ['Failed to connect to Weaviate server']
            }), 500
        
        logger.info("Getting Weaviate collection")
        # Get Weaviate collection
        collection = initialize_weaviate_collection(weaviate_client)
        if not collection:
            logger.error("Failed to initialize Weaviate collection")
            return jsonify({
                'success': False,
                'error': 'Failed to initialize Weaviate collection',
                'logs': ['Failed to initialize or access Weaviate collection']
            }), 500
        
        logger.info("Generating text embedding for query")
        # Generate a text embedding for the query
        query_embedding = get_text_embedding(query)
        
        logger.info("Searching Weaviate for similar items")
        # Search Weaviate for similar items
        results = collection.query.near_vector(
            near_vector=query_embedding.tolist(),
            limit=5
        ).do()
        
        logger.info(f"Connecting to S3 at: {s3_endpoint or 'default endpoint'}")
        # Get S3 client
        s3_client = get_s3_client(s3_endpoint, aws_access_key_id, aws_secret_access_key)
        
        # Process the results to include image data
        processed_results = []
        logs = []
        
        if results and 'data' in results and 'Get' in results['data'] and collection.name in results['data']['Get']:
            result_items = results['data']['Get'][collection.name]
            logger.info(f"Found {len(result_items)} matching results")
            logs.append(f"Found {len(result_items)} matching results in Weaviate")
            
            for idx, result in enumerate(result_items):
                try:
                    # Get the image from S3
                    s3_key = result.get('s3_key')
                    logs.append(f"Processing result {idx+1}: s3_key={s3_key}")
                    
                    # For the demo, we'll use placeholder images if S3 isn't configured
                    image_data = None
                    try:
                        logger.info(f"Retrieving image from S3: {s3_key}")
                        logs.append(f"Retrieving image from S3 bucket: {bucket_name}")
                        
                        image = read_image_from_s3(s3_client, bucket_name, s3_key)
                        if image:
                            logger.info(f"Successfully retrieved image: {s3_key}")
                            logs.append(f"Successfully retrieved image from S3")
                            
                            buffer = BytesIO()
                            image.save(buffer, format="JPEG")
                            image_data = buffer.getvalue()
                    except Exception as e:
                        error_msg = f"Error retrieving image from S3: {str(e)}"
                        logger.warning(error_msg)
                        logs.append(error_msg)
                        
                        # Use a placeholder image
                        placeholder_path = os.path.join('static', 'img', 'placeholder.jpg')
                        if os.path.exists(placeholder_path):
                            logger.info("Using placeholder image")
                            logs.append("Using placeholder image instead")
                            with open(placeholder_path, 'rb') as f:
                                image_data = f.read()
                    
                    if image_data:
                        # Convert image to base64 for embedding in HTML
                        img_str = base64.b64encode(image_data).decode('utf-8')
                        
                        processed_results.append({
                            's3_key': s3_key,
                            'text': result.get('text', 'No description available'),
                            'image': f"data:image/jpeg;base64,{img_str}"
                        })
                        logs.append(f"Result {idx+1} processed successfully")
                except Exception as e:
                    error_msg = f"Error processing result: {str(e)}"
                    logger.error(error_msg)
                    logs.append(error_msg)
        else:
            logger.warning("No results found or invalid response format")
            logs.append("No matching results found in Weaviate")
        
        return jsonify({
            'success': True,
            'results': processed_results,
            'logs': logs
        })
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': str(e),
            'logs': [error_msg]
        }), 500

@app.route('/api/embed', methods=['POST'])
def embed_data():
    try:
        logs = []
        
        # Get dataset from request
        dataset = request.json.get('dataset')
        if not isinstance(dataset, list):
            error_msg = "Invalid dataset format"
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': [error_msg]
            }), 400
        
        logger.info(f"Processing {len(dataset)} dataset items")
        logs.append(f"Processing {len(dataset)} dataset items")
        
        # Get S3 and Weaviate credentials from request
        weaviate_url = request.json.get('weaviate_url')
        weaviate_token = request.json.get('weaviate_token')
        
        # Connect to Weaviate
        logger.info(f"Connecting to Weaviate at {weaviate_url or 'default URL'}")
        logs.append(f"Connecting to Weaviate at {weaviate_url or 'default URL'}")
        
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token, verbose=True)
        
        # Check log file for recent entries if connection failed
        if not weaviate_client:
            try:
                with open('weaviate_connection.log', 'r') as log_file:
                    # Get the last 20 lines from the log file
                    log_lines = log_file.readlines()[-20:]
                    # Add these detailed logs to our response
                    detailed_logs = [line.strip() for line in log_lines]
                    logs.extend(["--- Detailed Connection Logs ---"] + detailed_logs)
            except Exception as log_err:
                logs.append(f"Note: Could not read detailed logs: {str(log_err)}")
                
            logger.error("Failed to connect to Weaviate")
            logs.append("Failed to connect to Weaviate")
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Weaviate',
                'logs': logs + ['Failed to connect to Weaviate server']
            }), 400
        
        # Get Weaviate collection
        logger.info("Initializing Weaviate collection")
        logs.append("Initializing Weaviate collection")
        
        collection = initialize_weaviate_collection(weaviate_client)
        if not collection:
            logger.error("Failed to initialize Weaviate collection")
            return jsonify({
                'success': False,
                'error': 'Failed to initialize Weaviate collection',
                'logs': logs + ['Failed to initialize or access Weaviate collection']
            }), 500
        
        # Process and embed each item in the dataset (text only)
        success_count = 0
        error_count = 0
        for idx, item in enumerate(dataset):
            try:
                s3_key = item.get('s3_key')
                text = item.get('text', 'No description')
                
                log_prefix = f"Item {idx+1}/{len(dataset)}"
                
                if not s3_key:
                    warning_msg = f"{log_prefix}: Skipping item without s3_key"
                    logger.warning(warning_msg)
                    logs.append(warning_msg)
                    error_count += 1
                    continue
                
                logger.info(f"{log_prefix}: Processing item with s3_key={s3_key}")
                logs.append(f"{log_prefix}: Processing item with s3_key={s3_key}")
                
                # Store in Weaviate using text embedding instead of image
                if store_text_embedding_in_weaviate(collection, s3_key, text):
                    success_msg = f"{log_prefix}: Successfully embedded"
                    logger.info(success_msg)
                    logs.append(success_msg)
                    success_count += 1
                else:
                    error_msg = f"{log_prefix}: Failed to embed"
                    logger.error(error_msg)
                    logs.append(error_msg)
                    error_count += 1
            except Exception as e:
                error_msg = f"Item {idx+1}: Error processing item: {str(e)}"
                logger.error(error_msg)
                logs.append(error_msg)
                error_count += 1
        
        result_msg = f'Text embedding complete. Successfully embedded {success_count} items. Failed: {error_count}.'
        logger.info(result_msg)
        logs.append(result_msg)
        
        return jsonify({
            'success': True,
            'message': result_msg,
            'logs': logs
        })
    except Exception as e:
        error_msg = f"Error in embed endpoint: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': str(e),
            'logs': [error_msg]
        }), 500

@app.route('/api/update-settings', methods=['POST'])
def update_settings():
    try:
        # Get settings from request
        settings = request.json
        
        # Test S3 connection with the new settings
        s3_client = get_s3_client(
            endpoint_url=settings.get('s3_endpoint'),
            aws_access_key_id=settings.get('aws_access_key_id'),
            aws_secret_access_key=settings.get('aws_secret_access_key')
        )
        
        # Try to list buckets to verify connection
        s3_client.list_buckets()
        
        # Test Weaviate connection if credentials provided
        if settings.get('weaviate_url'):
            weaviate_client = connect_to_weaviate(
                weaviate_url=settings.get('weaviate_url'),
                weaviate_token=settings.get('weaviate_token')
            )
            
            if not weaviate_client:
                return jsonify({
                    'success': False,
                    'error': 'Failed to connect to Weaviate with provided credentials'
                }), 400
        
        # Store settings in environment variables (for this session only)
        if settings.get('s3_endpoint'):
            os.environ['S3_ENDPOINT_URL'] = settings.get('s3_endpoint')
        if settings.get('aws_access_key_id'):
            os.environ['AWS_ACCESS_KEY_ID'] = settings.get('aws_access_key_id')
        if settings.get('aws_secret_access_key'):
            os.environ['AWS_SECRET_ACCESS_KEY'] = settings.get('aws_secret_access_key')
        if settings.get('s3_bucket_name'):
            os.environ['S3_BUCKET_NAME'] = settings.get('s3_bucket_name')
        if settings.get('weaviate_url'):
            os.environ['WEAVIATE_URL'] = settings.get('weaviate_url')
        if settings.get('weaviate_token'):
            os.environ['WEAVIATE_TOKEN'] = settings.get('weaviate_token')
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully'
        })
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pull-latest', methods=['POST'])
def pull_latest():
    try:
        # Get the directory where the application is running
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get repository URL from environment variable
        remote_url = os.getenv('GIT_REPO_URL')
        if not remote_url:
            return jsonify({
                'success': False,
                'error': 'GIT_REPO_URL environment variable not set',
                'logs': ['GIT_REPO_URL environment variable not set']
            })
        
        # Check if we have write permissions
        try:
            test_file = os.path.join(app_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            error_msg = f'No write permissions in application directory: {str(e)}'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': [error_msg]
            })

        logs = []
        
        # Make sure git status is clean first - check for any local changes
        status_result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=app_dir,
            capture_output=True,
            text=True
        )
        
        # Variable to track if we stashed changes
        stashed_changes = False
        
        if status_result.stdout.strip():
            # There are uncommitted changes, stash them
            log_msg = "Stashing local changes before pull"
            logger.info(log_msg)
            logs.append(log_msg)
            subprocess.run(['git', 'stash'], cwd=app_dir)
            stashed_changes = True
        
        # Get the current commit hash before pull
        before_pull_hash = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=app_dir,
            capture_output=True,
            text=True
        ).stdout.strip()
        
        # Get the remote hash to compare with local before pulling
        fetch_result = subprocess.run(
            ['git', 'fetch', '--quiet', 'origin', 'main'],
            cwd=app_dir,
            capture_output=True,
            text=True
        )
        
        if fetch_result.returncode != 0:
            error_msg = f'Git fetch failed: {fetch_result.stderr}'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': [error_msg]
            })
        
        # Check if there are any changes to pull
        remote_hash = subprocess.run(
            ['git', 'rev-parse', 'origin/main'],
            cwd=app_dir,
            capture_output=True,
            text=True
        ).stdout.strip()
        
        # If local is same as remote, no need to pull
        if before_pull_hash == remote_hash:
            log_message = "Already up to date. No changes to pull."
            logger.info(log_message)
            return jsonify({
                'success': True,
                'message': log_message,
                'needsRestart': False,
                'changedFiles': [],
                'templatesChanged': False,
                'requirementsChanged': False,
                'logs': [log_message]
            })
        
        # Run git pull
        result = subprocess.run(
            ['git', 'pull', 'origin', 'main'],
            cwd=app_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = f'Git pull failed: {result.stderr}'
            logger.error(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': [error_msg]
            })
        
        # Get the commit hash after pull
        after_pull_hash = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=app_dir,
            capture_output=True,
            text=True
        ).stdout.strip()
        
        # If we stashed changes earlier, apply them back now
        if stashed_changes:
            log_msg = "Applying stashed local changes after pull"
            logger.info(log_msg)
            logs.append(log_msg)
            stash_apply_result = subprocess.run(
                ['git', 'stash', 'pop'],
                cwd=app_dir,
                capture_output=True,
                text=True
            )
            if stash_apply_result.returncode != 0:
                conflict_msg = f"Warning: Could not apply stashed changes cleanly: {stash_apply_result.stderr}"
                logger.warning(conflict_msg)
                logs.append(conflict_msg)
        
        # Verify that the pull actually changed something
        if before_pull_hash == after_pull_hash:
            log_message = "Pull completed but no changes were made"
            logger.info(log_message)
            return jsonify({
                'success': True,
                'message': log_message,
                'needsRestart': False,
                'changedFiles': [],
                'templatesChanged': False,
                'requirementsChanged': False,
                'logs': [log_message]
            })
        
        # Check what files changed between the two commits
        changed_files = subprocess.run(
            ['git', 'diff', '--name-only', before_pull_hash, after_pull_hash],
            cwd=app_dir,
            capture_output=True,
            text=True
        ).stdout.splitlines()
        
        python_files_changed = any(f.endswith('.py') for f in changed_files)
        template_files_changed = any('templates/' in f for f in changed_files)
        static_files_changed = any('static/' in f for f in changed_files)
        requirements_changed = 'requirements.txt' in changed_files
        
        # Log the changes
        log_message = f"Changed files: {', '.join(changed_files)}" if changed_files else "No files changed"
        logger.info(log_message)
        logs.append(log_message)
        
        # Determine if we need a full restart or just a page refresh
        needs_restart = python_files_changed or requirements_changed
        needs_page_refresh = template_files_changed or static_files_changed
        
        response = {
            'success': True,
            'message': result.stdout,
            'needsRestart': needs_restart,
            'needsPageRefresh': needs_page_refresh,
            'changedFiles': changed_files,
            'templatesChanged': template_files_changed,
            'requirementsChanged': requirements_changed,
            'logs': logs
        }
        
        # If requirements.txt changed, install updated requirements
        if requirements_changed:
            req_log_msg = "requirements.txt changed, installing updated dependencies..."
            logger.info(req_log_msg)
            logs.append(req_log_msg)
            
            # Use a non-blocking approach to show status while pip runs
            try:
                # Run pip install -r requirements.txt
                pip_process = subprocess.Popen(
                    [sys.executable, '-m', 'pip', 'install', '-r', os.path.join(app_dir, 'requirements.txt')],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Get output with a timeout to avoid hanging
                try:
                    stdout, stderr = pip_process.communicate(timeout=60)
                    if pip_process.returncode != 0:
                        error_msg = f"Failed to install requirements: {stderr}"
                        logger.error(error_msg)
                        logs.append(error_msg)
                        response['success'] = False
                        response['error'] = error_msg
                    else:
                        success_msg = "Successfully installed updated requirements"
                        logger.info(success_msg)
                        logs.append(success_msg)
                except subprocess.TimeoutExpired:
                    pip_process.kill()
                    warning_msg = "Pip install is taking too long, proceeding with restart anyway"
                    logger.warning(warning_msg)
                    logs.append(warning_msg)
            except Exception as e:
                error_msg = f"Error during pip install: {str(e)}"
                logger.error(error_msg)
                logs.append(error_msg)
                # Continue with restart despite pip error - it might work
        
        # Update the logs in the response
        response['logs'] = logs
        
        if needs_restart:
            # If Python files or requirements changed, we need to restart
            logger.info("Python files or requirements changed, restarting application...")
            logs.append("Restarting application with new code/dependencies...")
            response['logs'] = logs
            
            # Return the response but don't restart yet - the client will call /api/restart
            return jsonify(response)
        elif needs_page_refresh:
            # Force refresh templates in memory
            logger.info("Template or static files changed, clearing template cache...")
            logs.append("Template or static files changed, clearing template cache...")
            app.jinja_env.cache = {}
            response['logs'] = logs
            
            # Return with indication that only page refresh is needed, not full restart
            return jsonify(response)
        else:
            # No restart or refresh needed
            logs.append("Changes detected, but no restart or refresh needed")
            response['logs'] = logs
            return jsonify(response)
        
    except Exception as e:
        error_msg = f"Error pulling latest changes: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': str(e),
            'logs': [error_msg]
        }), 500

@app.route('/api/restart', methods=['POST'])
def restart_app():
    try:
        cause = request.json.get('cause', 'unknown')
        logger.info(f"Received request to restart the application. Cause: {cause}")
        
        # Create a delayed restart to allow the response to be sent first
        def delayed_restart():
            time.sleep(1)  # Wait 1 second to allow the response to be sent
            logger.info("Executing delayed restart...")
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                logger.error(f"Error during restart: {str(e)}")
        
        # Start a background thread to restart the server
        import threading
        thread = threading.Thread(target=delayed_restart)
        thread.daemon = True
        thread.start()
        
        # Send the success response before restarting
        return jsonify({
            'success': True,
            'message': 'Restarting server...',
            'logs': ['Restart triggered, server will restart momentarily']
        })
    except Exception as e:
        error_msg = f"Error restarting application: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': str(e),
            'logs': [error_msg]
        }), 500

@app.route('/api/test-s3-connection', methods=['POST'])
def test_s3_connection():
    try:
        # Get S3 credentials from request
        s3_endpoint = request.json.get('s3_endpoint')
        aws_access_key_id = request.json.get('aws_access_key_id')
        aws_secret_access_key = request.json.get('aws_secret_access_key')
        bucket_name = request.json.get('s3_bucket_name')
        
        if not bucket_name:
            return jsonify({
                'success': False,
                'error': 'Bucket name is required',
                'logs': ['Bucket name is required for testing S3 connection']
            }), 400
        
        logs = [f"Attempting to connect to S3 at {s3_endpoint or 'default endpoint'}"]
        logger.info(logs[0])
        
        # Create S3 client
        s3_client = get_s3_client(s3_endpoint, aws_access_key_id, aws_secret_access_key)
        
        # Try to list the contents of the bucket to test connection
        try:
            log_msg = f"Testing access to bucket: {bucket_name}"
            logger.info(log_msg)
            logs.append(log_msg)
            
            response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
            
            if 'Contents' in response:
                log_msg = f"Successfully listed objects in bucket. Found {len(response['Contents'])} items."
                logger.info(log_msg)
                logs.append(log_msg)
                
                # Get one object to further verify access
                if len(response['Contents']) > 0:
                    first_key = response['Contents'][0]['Key']
                    log_msg = f"First object key: {first_key}"
                    logger.info(log_msg)
                    logs.append(log_msg)
            else:
                log_msg = "Connected to bucket but it appears to be empty"
                logger.info(log_msg)
                logs.append(log_msg)
            
            return jsonify({
                'success': True,
                'message': 'Successfully connected to S3 bucket',
                'logs': logs
            })
        except Exception as e:
            error_msg = f"Error accessing S3 bucket: {str(e)}"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
    except Exception as e:
        error_msg = f"Error testing S3 connection: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg,
            'logs': [error_msg]
        }), 500

@app.route('/api/test-weaviate-connection', methods=['POST'])
def test_weaviate_connection():
    try:
        # Get Weaviate credentials from request
        weaviate_url = request.json.get('weaviate_url')
        weaviate_token = request.json.get('weaviate_token')
        
        logs = [f"Attempting to connect to Weaviate at {weaviate_url or 'default URL'}"]
        logger.info(logs[0])
        
        # Connect to Weaviate with verbose logging enabled
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token, verbose=True)
        
        # Check log file for recent entries
        try:
            with open('weaviate_connection.log', 'r') as log_file:
                # Get the last 20 lines from the log file
                log_lines = log_file.readlines()[-20:]
                # Add these detailed logs to our response
                detailed_logs = [line.strip() for line in log_lines]
                logs.extend(["--- Detailed Connection Logs ---"] + detailed_logs)
        except Exception as log_err:
            logs.append(f"Note: Could not read detailed logs: {str(log_err)}")
        
        if not weaviate_client:
            error_msg = "Failed to connect to Weaviate"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
        
        # Test if Weaviate is ready
        if not weaviate_client.is_ready():
            error_msg = "Weaviate connection established but server is not ready"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
        
        # Get collection info to further verify connection (v4 API)
        try:
            # Get list of collections instead of schema (v4 API)
            collections = weaviate_client.collections.list_all()
            
            # Log information about collections
            num_collections = len(collections)
            log_msg = f"Weaviate contains {num_collections} collections"
            logger.info(log_msg)
            logs.append(log_msg)
            
            # List collection names
            # In v4, collections.list_all() already returns collection names as strings
            collection_names = collections  # No need to extract .name attribute
            if collection_names:
                logs.append(f"Collection names: {', '.join(collection_names)}")
            
            # Check if our collection exists
            collection_name = "MercedesImageEmbedding"
            collection_exists = collection_name in collection_names
            
            if collection_exists:
                logs.append(f"Collection '{collection_name}' exists")
                
                # Get a specific collection to check details
                collection = weaviate_client.collections.get(collection_name)
                
                # Get object count in collection
                count = collection.aggregate.over_all().with_meta_count().do()
                object_count = count.meta_count
                logs.append(f"Collection '{collection_name}' contains {object_count} objects")
            else:
                logs.append(f"Collection '{collection_name}' does not exist yet. It will be created when needed.")
            
            # Return success
            return jsonify({
                'success': True,
                'logs': logs
            })
        except Exception as e:
            error_msg = f"Error retrieving Weaviate collections: {str(e)}"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
    except Exception as e:
        error_msg = f"Error testing Weaviate connection: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg,
            'logs': [error_msg]
        }), 500

@app.route('/api/check-file', methods=['POST'])
def check_file():
    try:
        # Get configuration from request
        data = request.json
        s3_endpoint = data.get('s3_endpoint')
        aws_access_key_id = data.get('aws_access_key_id')
        aws_secret_access_key = data.get('aws_secret_access_key')
        bucket_name = data.get('s3_bucket_name')
        file_path = data.get('file_path', 'training/training.json')
        
        # Create S3 client
        s3_client = get_s3_client(s3_endpoint, aws_access_key_id, aws_secret_access_key)
        
        # Check if file exists
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            content = response['Body'].read().decode('utf-8')
            json_data = json.loads(content)
            
            return jsonify({
                'success': True,
                'message': f'File found with {len(json_data)} entries',
                'entry_count': len(json_data)
            })
        except Exception as e:
            logger.error(f"Error checking file: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    except Exception as e:
        logger.error(f"Error in check_file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Global variables to track embedding progress
embedding_status = {
    'status': 'idle',  # 'idle', 'in_progress', 'completed', 'error'
    'total': 0,
    'completed': 0,
    'error': None
}

@app.route('/api/start-embedding', methods=['POST'])
def start_embedding():
    try:
        global embedding_status
        
        # Get configuration from request
        data = request.json
        s3_endpoint = data.get('s3_endpoint')
        aws_access_key_id = data.get('aws_access_key_id')
        aws_secret_access_key = data.get('aws_secret_access_key')
        bucket_name = data.get('s3_bucket_name')
        file_path = data.get('file_path', 'training/training.json')
        weaviate_url = data.get('weaviate_url')
        weaviate_token = data.get('weaviate_token')
        
        # Create S3 client
        s3_client = get_s3_client(s3_endpoint, aws_access_key_id, aws_secret_access_key)
        
        # Connect to Weaviate
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token)
        if not weaviate_client:
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Weaviate'
            }), 400
        
        # Get the collection
        collection = initialize_weaviate_collection(weaviate_client)
        if not collection:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize Weaviate collection'
            }), 400
        
        # Get the JSON file
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
            content = response['Body'].read().decode('utf-8')
            json_data = json.loads(content)
            
            # Update status
            embedding_status = {
                'status': 'in_progress',
                'total': len(json_data),
                'completed': 0,
                'error': None
            }
            
            # Start embedding in a background thread
            import threading
            embedding_thread = threading.Thread(
                target=perform_embedding,
                args=(collection, json_data)
            )
            embedding_thread.daemon = True
            embedding_thread.start()
            
            return jsonify({
                'success': True,
                'message': f'Started embedding {len(json_data)} entries'
            })
        except Exception as e:
            logger.error(f"Error getting JSON file: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
    except Exception as e:
        logger.error(f"Error in start_embedding: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def perform_embedding(collection, json_data):
    """Process embedding in a background thread"""
    global embedding_status
    
    try:
        for i, item in enumerate(json_data):
            try:
                # Extract data from the JSON item
                s3_key = item.get('s3_key', '')
                text = item.get('text', '')
                
                # Store the text embedding
                success = store_text_embedding_in_weaviate(collection, s3_key, text)
                
                # Update progress only if successful
                if success:
                    embedding_status['completed'] += 1
                    logger.info(f"Embedded {embedding_status['completed']} of {embedding_status['total']}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error embedding item {i}: {str(e)}")
                # Continue with the next item
        
        # Update status to completed
        embedding_status['status'] = 'completed'
        logger.info("Embedding process completed")
    except Exception as e:
        logger.error(f"Error in embedding process: {str(e)}")
        embedding_status['status'] = 'error'
        embedding_status['error'] = str(e)

@app.route('/api/embedding-progress', methods=['GET'])
def embedding_progress():
    """Return the current embedding progress"""
    global embedding_status
    return jsonify(embedding_status)

@app.route('/api/list-s3-buckets', methods=['GET'])
def list_s3_buckets():
    """List all S3 buckets available to the configured credentials"""
    try:
        s3_client = get_s3_client()
        
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response.get('Buckets', [])]
        
        return jsonify({
            'success': True,
            'buckets': buckets
        })
    except Exception as e:
        logger.error(f"Error listing S3 buckets: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/list-s3-objects', methods=['GET'])
def list_s3_objects():
    """List objects in a specific S3 bucket with optional prefix"""
    try:
        bucket_name = request.args.get('bucket')
        prefix = request.args.get('prefix', '')
        delimiter = request.args.get('delimiter', '/')
        
        if not bucket_name:
            return jsonify({
                'success': False,
                'error': 'Bucket name is required'
            }), 400
            
        s3_client = get_s3_client()
        
        # List objects with pagination support
        params = {
            'Bucket': bucket_name,
            'Delimiter': delimiter,
            'MaxKeys': 1000
        }
        
        if prefix:
            params['Prefix'] = prefix
            
        response = s3_client.list_objects_v2(**params)
        
        # Process the response
        result = {
            'success': True,
            'prefix': prefix,
            'objects': [],
            'folders': [],
            'isTruncated': response.get('IsTruncated', False),
            'nextContinuationToken': response.get('NextContinuationToken', '')
        }
        
        # Extract folders (CommonPrefixes)
        for common_prefix in response.get('CommonPrefixes', []):
            folder_name = common_prefix.get('Prefix', '')
            result['folders'].append({
                'name': folder_name,
                'type': 'folder'
            })
            
        # Extract objects
        for content in response.get('Contents', []):
            # Skip the prefix itself if it's included as an object
            if content.get('Key') == prefix:
                continue
                
            # Get file size and last modified
            size = content.get('Size', 0)
            last_modified = content.get('LastModified').isoformat() if content.get('LastModified') else ''
            
            # Determine if this is a folder (ends with delimiter) or a file
            key = content.get('Key', '')
            is_folder = key.endswith(delimiter)
            
            # If it's a folder that wasn't already included in CommonPrefixes
            if is_folder:
                result['folders'].append({
                    'name': key,
                    'size': size,
                    'lastModified': last_modified,
                    'type': 'folder'
                })
            else:
                result['objects'].append({
                    'name': key,
                    'size': size,
                    'lastModified': last_modified,
                    'type': 'file'
                })
                
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error listing S3 objects: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/list-weaviate-schemas', methods=['GET'])
def list_weaviate_schemas():
    """List all schemas/collections in Weaviate"""
    try:
        weaviate_client = connect_to_weaviate()
        
        if not weaviate_client:
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Weaviate'
            }), 400
            
        # Get schema details
        schema = weaviate_client.schema.get()
        
        # Extract classes (collections)
        collections = []
        for class_obj in schema.get('classes', []):
            class_name = class_obj.get('class')
            
            # Get object count for this class
            try:
                count = weaviate_client.query.aggregate(class_name).with_meta_count().do()
                object_count = count.get('data', {}).get('Aggregate', {}).get(class_name, [{}])[0].get('meta', {}).get('count', 0)
            except Exception as count_err:
                logger.warning(f"Error getting count for {class_name}: {str(count_err)}")
                object_count = -1
                
            collections.append({
                'name': class_name,
                'description': class_obj.get('description', ''),
                'vectorIndexType': class_obj.get('vectorIndexType', ''),
                'vectorizer': class_obj.get('vectorizer', ''),
                'count': object_count
            })
            
        return jsonify({
            'success': True,
            'collections': collections
        })
    except Exception as e:
        logger.error(f"Error listing Weaviate schemas: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/list-weaviate-objects', methods=['GET'])
def list_weaviate_objects():
    """List objects in a specific Weaviate collection with pagination"""
    try:
        collection_name = request.args.get('collection')
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))
        
        if not collection_name:
            return jsonify({
                'success': False,
                'error': 'Collection name is required'
            }), 400
            
        weaviate_client = connect_to_weaviate()
        
        if not weaviate_client:
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Weaviate'
            }), 400
            
        # Get total count first
        try:
            count_result = weaviate_client.query.aggregate(collection_name).with_meta_count().do()
            total_count = count_result.get('data', {}).get('Aggregate', {}).get(collection_name, [{}])[0].get('meta', {}).get('count', 0)
        except Exception as count_err:
            logger.warning(f"Error getting count for {collection_name}: {str(count_err)}")
            total_count = -1
            
        # Now get objects with pagination
        try:
            query_result = (
                weaviate_client.query
                .get(collection_name, ['id', 'image_url', 'description'])
                .with_limit(limit)
                .with_offset(offset)
                .do()
            )
            
            objects = query_result.get('data', {}).get('Get', {}).get(collection_name, [])
        except Exception as query_err:
            logger.error(f"Error querying objects in {collection_name}: {str(query_err)}")
            return jsonify({
                'success': False,
                'error': f'Error querying objects: {str(query_err)}'
            }), 500
            
        return jsonify({
            'success': True,
            'collection': collection_name,
            'objects': objects,
            'pagination': {
                'total': total_count,
                'offset': offset,
                'limit': limit
            }
        })
    except Exception as e:
        logger.error(f"Error listing Weaviate objects: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/save-settings', methods=['POST'])
def save_settings():
    """Save settings to environmental variables and optionally to .env file"""
    try:
        # Get settings from request
        settings = request.json
        persistToFile = settings.pop('persistToFile', False)
        
        # Update environment variables for this session
        if settings.get('s3_endpoint'):
            os.environ['S3_ENDPOINT_URL'] = settings.get('s3_endpoint')
        if settings.get('aws_access_key_id'):
            os.environ['AWS_ACCESS_KEY_ID'] = settings.get('aws_access_key_id')
        if settings.get('aws_secret_access_key'):
            os.environ['AWS_SECRET_ACCESS_KEY'] = settings.get('aws_secret_access_key')
        if settings.get('s3_bucket_name'):
            os.environ['S3_BUCKET_NAME'] = settings.get('s3_bucket_name')
        if settings.get('weaviate_url'):
            os.environ['WEAVIATE_URL'] = settings.get('weaviate_url')
        if settings.get('weaviate_token'):
            os.environ['WEAVIATE_TOKEN'] = settings.get('weaviate_token')
            
        # Optionally save to .env file for persistence
        if persistToFile:
            try:
                # Read existing .env file if it exists
                env_data = {}
                if os.path.exists('.env'):
                    with open('.env', 'r') as env_file:
                        for line in env_file:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            key, value = line.split('=', 1)
                            env_data[key] = value
                
                # Update with new values
                if settings.get('s3_endpoint'):
                    env_data['S3_ENDPOINT_URL'] = settings.get('s3_endpoint')
                if settings.get('aws_access_key_id'):
                    env_data['AWS_ACCESS_KEY_ID'] = settings.get('aws_access_key_id')
                if settings.get('aws_secret_access_key'):
                    env_data['AWS_SECRET_ACCESS_KEY'] = settings.get('aws_secret_access_key')
                if settings.get('s3_bucket_name'):
                    env_data['S3_BUCKET_NAME'] = settings.get('s3_bucket_name')
                if settings.get('weaviate_url'):
                    env_data['WEAVIATE_URL'] = settings.get('weaviate_url')
                if settings.get('weaviate_token'):
                    env_data['WEAVIATE_TOKEN'] = settings.get('weaviate_token')
                
                # Write back to .env file
                with open('.env', 'w') as env_file:
                    for key, value in env_data.items():
                        env_file.write(f"{key}={value}\n")
                
                logger.info("Settings saved to .env file")
            except Exception as file_err:
                logger.error(f"Error saving settings to .env file: {str(file_err)}")
                return jsonify({
                    'success': True,
                    'message': 'Settings updated for this session only. Failed to save to .env file.',
                    'fileError': str(file_err)
                })
        
        return jsonify({
            'success': True,
            'message': 'Settings updated successfully' + (' and saved to .env file' if persistToFile else ' for this session')
        })
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)