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

# Embedding model setup
def load_embedding_model():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

# Initialize the embedding model
embedding_model = load_embedding_model()

# Weaviate client setup
def connect_to_weaviate(weaviate_url=None, weaviate_token=None):
    """Connect to Weaviate instance using provided credentials or environment variables"""
    url = weaviate_url or os.getenv('WEAVIATE_URL', 'weaviate.poc-weaviate.svc.cluster.local')
    token = weaviate_token or os.getenv('WEAVIATE_TOKEN')
    
    headers = {}
    if token:
        headers["x-auth-token"] = token
    
    try:
        client = weaviate.connect_to_custom(
            http_host=url,
            http_port=80,
            http_secure=False,
            grpc_host=f"{url}-grpc.poc-weaviate.svc.cluster.local" if '-grpc' not in url else url,
            grpc_port=50051,
            grpc_secure=False,
            headers=headers,
            skip_init_checks=False
        )
        
        if client.is_ready():
            logger.info("Successfully connected to Weaviate")
            return client
        else:
            logger.error("Failed to connect to Weaviate")
            return None
    except Exception as e:
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
    """Generate text embedding using a suitable text embedding model"""
    try:
        # For simplicity, we're using a random embedding for demonstration
        # In a production environment, you'd use a proper text embedding model
        # Generate a random embedding vector of size 1536 (suitable for OpenAI-like models)
        embedding = np.random.rand(1536).astype(np.float32)
        # Normalize the embedding to have unit norm (common practice)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    except Exception as e:
        logger.error(f"Error generating text embedding: {str(e)}")
        return None

def store_text_embedding_in_weaviate(collection, s3_key, text):
    """Store a text and its embedding in Weaviate"""
    try:
        # Generate embedding for the text
        embedding = get_text_embedding(text)
        
        if embedding is None:
            raise ValueError("Failed to generate embedding for text")
        
        # Store in Weaviate
        collection.data.insert(
            properties={
                "s3_key": s3_key,
                "text": text
            },
            vector=embedding.tolist()  # Convert numpy array to list
        )
        
        return True
    except Exception as e:
        logger.error(f"Error storing text embedding in Weaviate: {str(e)}")
        raise

@app.route('/')
def index():
    response = render_template('index.html')
    # Add cache control headers to prevent browser caching
    return response, {'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'}

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

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
        # Get Weaviate credentials from request
        weaviate_url = request.json.get('weaviate_url')
        weaviate_token = request.json.get('weaviate_token')
        dataset = request.json.get('dataset', [])
        
        logs = []
        
        # Validate dataset
        if not dataset or not isinstance(dataset, list):
            logger.error("Invalid dataset format")
            return jsonify({
                'success': False,
                'error': 'Invalid dataset format. Expected a list of items with s3_key and text fields.',
                'logs': ['Invalid dataset format. Expected a list of items with s3_key and text fields.']
            }), 400
        
        logger.info(f"Processing {len(dataset)} dataset items")
        logs.append(f"Processing {len(dataset)} dataset items for embedding")
        
        # Connect to Weaviate
        logger.info(f"Connecting to Weaviate at {weaviate_url or 'default URL'}")
        logs.append(f"Connecting to Weaviate at {weaviate_url or 'default URL'}")
        
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token)
        if not weaviate_client:
            logger.error("Failed to connect to Weaviate")
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Weaviate',
                'logs': logs + ['Failed to connect to Weaviate server']
            }), 500
        
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

        # Stash any local changes
        subprocess.run(['git', 'stash'], cwd=app_dir)
        
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
        
        # Check what files changed
        changed_files = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD@{1}', 'HEAD'],
            cwd=app_dir,
            capture_output=True,
            text=True
        ).stdout.splitlines()
        
        python_files_changed = any(f.endswith('.py') for f in changed_files)
        template_files_changed = any('templates/' in f for f in changed_files)
        requirements_changed = 'requirements.txt' in changed_files
        
        # Log the changes
        log_message = f"Changed files: {', '.join(changed_files)}" if changed_files else "No files changed"
        logger.info(log_message)
        
        logs = [log_message]
        
        response = {
            'success': True,
            'message': result.stdout,
            'needsRestart': python_files_changed or requirements_changed,
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
        
        if python_files_changed or requirements_changed:
            # If Python files or requirements changed, we need to restart
            logger.info("Python files or requirements changed, restarting application...")
            logs.append("Restarting application with new code/dependencies...")
            response['logs'] = logs
            
            # Return the response but don't restart yet - the client will call /api/restart
            return jsonify(response)
        elif template_files_changed:
            # Force refresh templates in memory
            logger.info("Template files changed, clearing template cache...")
            logs.append("Template files changed, clearing template cache...")
            app.jinja_env.cache = {}
        
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
        
        # Connect to Weaviate
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token)
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
        
        # Get schema info to further verify connection
        try:
            schema = weaviate_client.schema.get()
            
            # Log information about schema
            if 'classes' in schema:
                num_classes = len(schema['classes'])
                log_msg = f"Schema contains {num_classes} classes"
                logger.info(log_msg)
                logs.append(log_msg)
                
                # Check if our collection exists
                collection_exists = False
                for cls in schema['classes']:
                    if cls['class'] == 'MercedesImageEmbedding':
                        collection_exists = True
                        log_msg = f"Found MercedesImageEmbedding collection"
                        logger.info(log_msg)
                        logs.append(log_msg)
                        break
                
                if not collection_exists:
                    log_msg = "MercedesImageEmbedding collection not found, will be created when needed"
                    logger.info(log_msg)
                    logs.append(log_msg)
            else:
                log_msg = "Connected to Weaviate but schema appears to be empty"
                logger.info(log_msg)
                logs.append(log_msg)
            
            return jsonify({
                'success': True,
                'message': 'Successfully connected to Weaviate',
                'logs': logs
            })
        except Exception as e:
            error_msg = f"Error retrieving Weaviate schema: {str(e)}"
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

@app.route('/api/retrieve-training-file', methods=['POST'])
def retrieve_training_file():
    try:
        # Get S3 credentials from request
        s3_endpoint = request.json.get('s3_endpoint') or os.getenv('S3_ENDPOINT_URL')
        aws_access_key_id = request.json.get('aws_access_key_id') or os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = request.json.get('aws_secret_access_key') or os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket_name = request.json.get('s3_bucket_name') or os.getenv('S3_BUCKET_NAME')
        
        if not bucket_name:
            return jsonify({
                'success': False,
                'error': 'Bucket name is required',
                'logs': ['Bucket name is required for retrieving training file']
            }), 400
        
        logs = ["Attempting to retrieve training.json file from S3"]
        logger.info(logs[0])
        
        # Create S3 client
        s3_client = get_s3_client(s3_endpoint, aws_access_key_id, aws_secret_access_key)
        
        # Try to get the training.json file
        try:
            file_key = "training.json"
            log_msg = f"Retrieving file: {file_key} from bucket: {bucket_name}"
            logger.info(log_msg)
            logs.append(log_msg)
            
            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            file_content = response['Body'].read().decode('utf-8')
            training_data = json.loads(file_content)
            
            log_msg = f"Successfully retrieved training file with {len(training_data)} entries"
            logger.info(log_msg)
            logs.append(log_msg)
            
            # Store the training data in the session for later use
            session_key = f"training_data_{int(time.time())}"
            app.config[session_key] = training_data
            
            return jsonify({
                'success': True,
                'message': 'Successfully retrieved training.json file',
                'count': len(training_data),
                'session_key': session_key,
                'logs': logs
            })
        except Exception as e:
            error_msg = f"Error retrieving training file: {str(e)}"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
    except Exception as e:
        error_msg = f"Error retrieving training file: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg,
            'logs': [error_msg]
        }), 500

@app.route('/api/embed-training-data', methods=['POST'])
def embed_training_data():
    try:
        # Get the session key for the training data
        session_key = request.json.get('session_key')
        if not session_key or session_key not in app.config:
            return jsonify({
                'success': False,
                'error': 'Training data not found in session',
                'logs': ['Please retrieve the training file first']
            }), 400
        
        # Get Weaviate credentials from request
        weaviate_url = request.json.get('weaviate_url') or os.getenv('WEAVIATE_URL')
        weaviate_token = request.json.get('weaviate_token') or os.getenv('WEAVIATE_TOKEN')
        
        logs = ["Starting to embed training data in Weaviate"]
        logger.info(logs[0])
        
        # Connect to Weaviate
        weaviate_client = connect_to_weaviate(weaviate_url, weaviate_token)
        if not weaviate_client:
            error_msg = "Failed to connect to Weaviate"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
        
        # Initialize collection if needed
        collection = initialize_weaviate_collection(weaviate_client)
        if not collection:
            error_msg = "Failed to initialize Weaviate collection"
            logger.error(error_msg)
            logs.append(error_msg)
            return jsonify({
                'success': False,
                'error': error_msg,
                'logs': logs
            }), 400
        
        # Get the training data
        training_data = app.config[session_key]
        
        # Embed each entry
        successful = 0
        failed = 0
        
        for i, entry in enumerate(training_data):
            try:
                s3_key = entry.get('s3_key')
                text = entry.get('text')
                
                if not s3_key or not text:
                    log_msg = f"Skipping entry {i}: Missing s3_key or text"
                    logger.warning(log_msg)
                    logs.append(log_msg)
                    failed += 1
                    continue
                
                # Get text embedding and store in Weaviate
                store_text_embedding_in_weaviate(collection, s3_key, text)
                
                successful += 1
                
                # Log progress periodically
                if successful % 10 == 0:
                    log_msg = f"Embedded {successful} entries so far"
                    logger.info(log_msg)
                    logs.append(log_msg)
            except Exception as e:
                failed += 1
                error_msg = f"Error embedding entry {i}: {str(e)}"
                logger.error(error_msg)
                logs.append(error_msg)
        
        final_msg = f"Embedding complete. Successfully embedded {successful} entries. Failed: {failed}"
        logger.info(final_msg)
        logs.append(final_msg)
        
        return jsonify({
            'success': True,
            'message': final_msg,
            'successful': successful,
            'failed': failed,
            'logs': logs
        })
    except Exception as e:
        error_msg = f"Error embedding training data: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg,
            'logs': [error_msg]
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)