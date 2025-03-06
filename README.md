# Mercedes Image Chatbot

A simple chatbot application that searches for images based on textual descriptions. This application uses a vector database to find relevant images based on user queries and displays them in a carousel.

## Features

- Search for images using natural language queries
- Vector database search to find semantically similar content
- Image retrieval from S3 bucket using boto3
- Interactive image carousel display
- Configurable S3 connection settings
- Git-based application update mechanism

## How It Works

1. User enters a description of the images they're looking for
2. The application searches a vector database containing image caption embeddings
3. Matching image keys are retrieved from the vector database
4. The application uses boto3 to fetch the actual images from an S3 bucket
5. Images and their captions are displayed in a carousel

## Setup

### Prerequisites

- Python 3.7+
- S3-compatible storage with appropriate credentials
- Vector database with embedded image descriptions

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd MercedesChat
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at `http://localhost:8080`

### Configuration

The application can be configured using environment variables or through the settings panel in the user interface:

- `S3_ENDPOINT_URL`: URL endpoint of your S3-compatible storage
- `AWS_ACCESS_KEY_ID`: Access key for S3
- `AWS_SECRET_ACCESS_KEY`: Secret key for S3
- `S3_BUCKET_NAME`: Name of the bucket containing images
- `GIT_REPO_URL`: URL of the repository for auto-updates

## Vector Database Format

The application expects a vector database with entries in the following format:

```python
{
    's3_key': 'path/to/image.jpg',
    'text': 'A description of the image content',
    'vector': [0.1, 0.2, 0.3, ...]  # Embedding vector
}
```

## Architecture

- Flask web application
- FAISS for vector similarity search
- boto3 for S3 interaction
- JavaScript frontend with dynamic image loading

## License

This project is licensed under the MIT License - see the LICENSE file for details.
