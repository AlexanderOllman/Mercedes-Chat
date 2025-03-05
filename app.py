from flask import Flask, render_template, jsonify
import subprocess
import os
import sys
import logging

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

@app.route('/')
def index():
    response = render_template('index.html')
    # Add cache control headers to prevent browser caching
    return response, {'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'}

@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})

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