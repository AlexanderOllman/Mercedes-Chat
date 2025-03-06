from flask import Flask, render_template, jsonify, request
import subprocess
import os
import sys
import logging
import json
import boto3
from io import BytesIO
import base64
from PIL import Image
import faiss
import numpy as np
from dotenv import load_dotenv

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

# Vector database setup
class VectorDatabase:
    def __init__(self):
        self.index = None
        self.data = []
        self.dimension = 128  # Example dimension, adjust based on your embedding size
        
    def initialize(self):
        """Initialize an empty FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info("Vector database initialized")
        
    def load_data(self, data_path=None):
        """Load sample data or use provided data path"""
        # For testing, we'll use some sample data
        if not data_path:
            self.data = [
                {'s3_key': 'sample/image1.jpg', 'text': 'A photo of a beach', 'vector': np.random.random(self.dimension).astype('float32')},
                {'s3_key': 'sample/image2.jpg', 'text': 'A photo of a mountain', 'vector': np.random.random(self.dimension).astype('float32')},
                {'s3_key': 'sample/image3.jpg', 'text': 'A photo of a floor', 'vector': np.random.random(self.dimension).astype('float32')},
                {'s3_key': 'sample/image4.jpg', 'text': 'A photo of a car', 'vector': np.random.random(self.dimension).astype('float32')},
                {'s3_key': 'sample/image5.jpg', 'text': 'A photo of a building', 'vector': np.random.random(self.dimension).astype('float32')}
            ]
        else:
            # Load from file
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        
        # Add vectors to index
        vectors = np.array([item['vector'] for item in self.data]).astype('float32')
        self.index.add(vectors)
        logger.info(f"Loaded {len(self.data)} items into vector database")
    
    def search(self, query_vector, top_k=5):
        """Search for similar items"""
        # In a real app, you would compute embeddings for the query
        # For demo, we'll just use a random vector
        query_vector = np.random.random(self.dimension).astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Return the results
        results = [self.data[idx] for idx in indices[0]]
        return results

# Initialize the vector database
vector_db = VectorDatabase()
vector_db.initialize()
vector_db.load_data()

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
        logger.info(f"Received query: {query}")
        
        # Search the vector database
        results = vector_db.search(query, top_k=5)
        
        # Get S3 credentials from request if available
        endpoint_url = request.json.get('s3_endpoint')
        aws_access_key_id = request.json.get('aws_access_key_id')
        aws_secret_access_key = request.json.get('aws_secret_access_key')
        
        # Get S3 client
        s3_client = get_s3_client(endpoint_url, aws_access_key_id, aws_secret_access_key)
        
        # Process the results to include image data
        processed_results = []
        for result in results:
            try:
                # Get the image from S3
                s3_key = result['s3_key']
                bucket = os.getenv('S3_BUCKET_NAME', 'default-bucket')
                
                # For the demo, we'll use placeholder images if S3 isn't configured
                image_data = None
                try:
                    response = s3_client.get_object(Bucket=bucket, Key=s3_key)
                    image_data = response['Body'].read()
                except Exception as e:
                    logger.warning(f"Error retrieving image from S3: {str(e)}")
                    # Use a placeholder image
                    placeholder_path = os.path.join('static', 'img', 'placeholder.jpg')
                    if os.path.exists(placeholder_path):
                        with open(placeholder_path, 'rb') as f:
                            image_data = f.read()
                
                if image_data:
                    # Convert image to base64 for embedding in HTML
                    try:
                        # Try to open with PIL to handle format conversion if needed
                        img = Image.open(BytesIO(image_data))
                        buffer = BytesIO()
                        img.save(buffer, format="JPEG")
                        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        processed_results.append({
                            's3_key': s3_key,
                            'text': result['text'],
                            'image': f"data:image/jpeg;base64,{img_str}"
                        })
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing result {result}: {str(e)}")
        
        return jsonify({
            'success': True,
            'results': processed_results
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
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
        
        # Store settings in environment variables (for this session only)
        if settings.get('s3_endpoint'):
            os.environ['S3_ENDPOINT_URL'] = settings.get('s3_endpoint')
        if settings.get('aws_access_key_id'):
            os.environ['AWS_ACCESS_KEY_ID'] = settings.get('aws_access_key_id')
        if settings.get('aws_secret_access_key'):
            os.environ['AWS_SECRET_ACCESS_KEY'] = settings.get('aws_secret_access_key')
        if settings.get('s3_bucket_name'):
            os.environ['S3_BUCKET_NAME'] = settings.get('s3_bucket_name')
        
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
                'error': 'GIT_REPO_URL environment variable not set'
            })
        
        # Check if we have write permissions
        try:
            test_file = os.path.join(app_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'No write permissions in application directory: {str(e)}'
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
            return jsonify({
                'success': False,
                'error': f'Git pull failed: {result.stderr}'
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
        
        response = {
            'success': True,
            'message': result.stdout,
            'needsRestart': python_files_changed,
            'changedFiles': changed_files,
            'templatesChanged': template_files_changed
        }
        
        if python_files_changed:
            # If Python files changed, we need to restart
            # Use os.execv to restart the current process
            logger.info("Python files changed, restarting application...")
            os.execv(sys.executable, ['python'] + sys.argv)
        elif template_files_changed:
            # Force refresh templates in memory
            logger.info("Template files changed, clearing template cache...")
            app.jinja_env.cache = {}
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error pulling latest changes: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)