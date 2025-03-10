from flask import Flask, render_template, jsonify, request
import os
import logging
import json
import boto3
from io import BytesIO
import base64
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import weaviate

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

# Create a specific logger for Weaviate connections
weaviate_logger = logging.getLogger('weaviate_connection')
weaviate_logger.setLevel(logging.DEBUG)

# Add a file handler for Weaviate logs
weaviate_file_handler = logging.FileHandler('weaviate_connection.log')
weaviate_file_handler.setLevel(logging.DEBUG)
weaviate_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
weaviate_logger.addHandler(weaviate_file_handler)

# S3 client setup
def get_s3_client(endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None):
    """Connect to S3 using provided credentials or environment variables"""
    endpoint_url = endpoint_url or os.getenv('S3_ENDPOINT_URL')
    aws_access_key_id = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
    
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    return client

# Weaviate client setup
def connect_to_weaviate(weaviate_url=None, weaviate_token=None, verbose=False):
    """Connect to Weaviate instance using provided credentials or environment variables"""
    if verbose:
        weaviate_logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        weaviate_logger.addHandler(console_handler)
    
    url = weaviate_url or os.getenv('WEAVIATE_URL', 'weaviate.poc-weaviate.svc.cluster.local')
    token = weaviate_token or os.getenv('WEAVIATE_TOKEN')
    
    weaviate_logger.debug(f"Attempting to connect to Weaviate at: {url}")
    
    headers = {}
    if token:
        headers = {"x-auth-token": token}
    
    try:
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
        return client
    except Exception as e:
        weaviate_logger.error(f"Failed to connect to Weaviate: {str(e)}")
        return None

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({"status": "ok"})

# S3 Connections and Management
@app.route('/api/test-s3-connection', methods=['POST'])
def test_s3_connection():
    """Test connection to S3 with provided credentials"""
    data = request.json
    
    endpoint_url = data.get('endpoint_url')
    aws_access_key_id = data.get('aws_access_key_id') 
    aws_secret_access_key = data.get('aws_secret_access_key')
    
    try:
        s3_client = get_s3_client(
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        # Test by listing buckets
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        return jsonify({
            "success": True,
            "message": f"Successfully connected to S3. Found {len(buckets)} buckets.",
            "buckets": buckets
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to connect to S3: {str(e)}"
        }), 400

@app.route('/api/list-s3-buckets', methods=['GET'])
def list_s3_buckets():
    """List all S3 buckets"""
    try:
        s3_client = get_s3_client()
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        return jsonify({
            "success": True,
            "buckets": buckets
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to list S3 buckets: {str(e)}"
        }), 400

@app.route('/api/list-s3-objects', methods=['GET'])
def list_s3_objects():
    """List objects in an S3 bucket with optional prefix"""
    bucket = request.args.get('bucket')
    prefix = request.args.get('prefix', '')
    
    if not bucket:
        return jsonify({
            "success": False,
            "message": "Bucket name is required"
        }), 400
    
    try:
        s3_client = get_s3_client()
        
        # Normalize the prefix to ensure proper directory listing
        if prefix and not prefix.endswith('/'):
            prefix = prefix + '/'
        
        # List objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
        
        result = {
            "success": True,
            "directories": [],
            "files": []
        }
        
        for page in pages:
            # Get common prefixes (directories)
            for prefix_dict in page.get('CommonPrefixes', []):
                prefix_path = prefix_dict.get('Prefix', '')
                # Remove the trailing slash and get the last component
                if prefix_path.endswith('/'):
                    prefix_path = prefix_path[:-1]
                dir_name = prefix_path.split('/')[-1]
                result["directories"].append({
                    "name": dir_name,
                    "path": prefix_dict.get('Prefix', '')
                })
            
            # Get files
            for content in page.get('Contents', []):
                key = content.get('Key', '')
                # Skip the directory marker itself
                if key != prefix and not key.endswith('/'):
                    file_name = key.split('/')[-1]
                    last_modified = content.get('LastModified', '')
                    if last_modified:
                        last_modified = last_modified.strftime('%Y-%m-%d %H:%M:%S')
                    result["files"].append({
                        "name": file_name,
                        "path": key,
                        "size": content.get('Size', 0),
                        "last_modified": last_modified
                    })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to list S3 objects: {str(e)}"
        }), 400

# Weaviate Connections and Management
@app.route('/api/test-weaviate-connection', methods=['POST'])
def test_weaviate_connection():
    """Test connection to Weaviate with provided credentials"""
    data = request.json
    
    weaviate_url = data.get('weaviate_url')
    weaviate_token = data.get('weaviate_token')
    
    try:
        client = connect_to_weaviate(
            weaviate_url=weaviate_url,
            weaviate_token=weaviate_token,
            verbose=True
        )
        
        if not client:
            return jsonify({
                "success": False,
                "message": "Failed to connect to Weaviate. Check logs for details."
            }), 400
        
        # Test by getting schema
        schema = client.schema.get()
        collection_names = [cls.get('class') for cls in schema.get('classes', [])]
        
        return jsonify({
            "success": True,
            "message": f"Successfully connected to Weaviate. Found {len(collection_names)} collections.",
            "collections": collection_names
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to connect to Weaviate: {str(e)}"
        }), 400

@app.route('/api/list-weaviate-collections', methods=['GET'])
def list_weaviate_collections():
    """List all Weaviate collections (schemas)"""
    try:
        client = connect_to_weaviate()
        
        if not client:
            return jsonify({
                "success": False,
                "message": "Failed to connect to Weaviate. Check logs for details."
            }), 400
        
        schema = client.schema.get()
        collections = []
        
        for cls in schema.get('classes', []):
            class_name = cls.get('class')
            properties = cls.get('properties', [])
            
            collection_info = {
                "name": class_name,
                "description": cls.get('description', ''),
                "vector_index_type": cls.get('vectorIndexConfig', {}).get('distance', 'cosine'),
                "property_count": len(properties),
                "properties": [prop.get('name') for prop in properties[:5]]  # Show first 5 properties only
            }
            
            # Get collection stats
            try:
                count = client.collections.get(class_name).aggregate.over_all().with_meta_count().do()
                collection_info["object_count"] = count["meta"]["count"]
            except:
                collection_info["object_count"] = "Unknown"
                
            collections.append(collection_info)
        
        return jsonify({
            "success": True,
            "collections": collections
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to list Weaviate collections: {str(e)}"
        }), 400

@app.route('/api/create-weaviate-collection', methods=['POST'])
def create_weaviate_collection():
    """Create a new Weaviate collection"""
    data = request.json
    
    collection_name = data.get('name')
    description = data.get('description', '')
    properties = data.get('properties', [])
    vector_index_config = data.get('vector_index_config', {
        "distance": "cosine"
    })
    
    if not collection_name:
        return jsonify({
            "success": False,
            "message": "Collection name is required"
        }), 400
    
    try:
        client = connect_to_weaviate()
        
        if not client:
            return jsonify({
                "success": False,
                "message": "Failed to connect to Weaviate. Check logs for details."
            }), 400
        
        # Check if collection already exists
        schema = client.schema.get()
        existing_classes = [cls.get('class') for cls in schema.get('classes', [])]
        
        if collection_name in existing_classes:
            return jsonify({
                "success": False,
                "message": f"Collection '{collection_name}' already exists"
            }), 400
        
        # Create collection class
        class_obj = {
            "class": collection_name,
            "description": description,
            "vectorIndexConfig": vector_index_config
        }
        
        if properties:
            class_obj["properties"] = properties
        
        client.schema.create_class(class_obj)
        
        return jsonify({
            "success": True,
            "message": f"Collection '{collection_name}' created successfully"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to create Weaviate collection: {str(e)}"
        }), 400

@app.route('/api/delete-weaviate-collection', methods=['POST'])
def delete_weaviate_collection():
    """Delete a Weaviate collection"""
    data = request.json
    
    collection_name = data.get('name')
    
    if not collection_name:
        return jsonify({
            "success": False,
            "message": "Collection name is required"
        }), 400
    
    try:
        client = connect_to_weaviate()
        
        if not client:
            return jsonify({
                "success": False,
                "message": "Failed to connect to Weaviate. Check logs for details."
            }), 400
        
        # Check if collection exists
        schema = client.schema.get()
        existing_classes = [cls.get('class') for cls in schema.get('classes', [])]
        
        if collection_name not in existing_classes:
            return jsonify({
                "success": False,
                "message": f"Collection '{collection_name}' does not exist"
            }), 400
        
        # Delete collection
        client.schema.delete_class(collection_name)
        
        return jsonify({
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to delete Weaviate collection: {str(e)}"
        }), 400

@app.route('/api/list-weaviate-objects', methods=['GET'])
def list_weaviate_objects():
    """List objects in a Weaviate collection with pagination"""
    collection_name = request.args.get('collection')
    limit = int(request.args.get('limit', 20))
    offset = int(request.args.get('offset', 0))
    
    if not collection_name:
        return jsonify({
            "success": False,
            "message": "Collection name is required"
        }), 400
    
    try:
        client = connect_to_weaviate()
        
        if not client:
            return jsonify({
                "success": False,
                "message": "Failed to connect to Weaviate. Check logs for details."
            }), 400
        
        # Get objects from collection with pagination
        collection = client.collections.get(collection_name)
        
        # Get total count first
        count_result = collection.aggregate.over_all().with_meta_count().do()
        total_count = count_result["meta"]["count"]
        
        # Then get objects with pagination
        query_result = collection.query.fetch_objects(
            limit=limit,
            offset=offset,
            include_vector=False
        )
        
        objects = []
        for obj in query_result.objects:
            properties = obj.properties
            obj_info = {
                "id": obj.uuid,
                "properties": properties
            }
            objects.append(obj_info)
        
        return jsonify({
            "success": True,
            "collection": collection_name,
            "total_count": total_count,
            "objects": objects,
            "limit": limit,
            "offset": offset
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Failed to list Weaviate objects: {str(e)}"
        }), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 