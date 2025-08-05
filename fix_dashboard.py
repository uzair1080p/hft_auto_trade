# fix_dashboard.py

"""
Script to fix dashboard connection issues and clear cached data.
"""

import os
import shutil
import subprocess
import sys

def clear_streamlit_cache():
    """Clear Streamlit cache directories."""
    cache_dirs = [
        os.path.expanduser("~/.streamlit"),
        ".streamlit",
        "__pycache__",
        ".pytest_cache"
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Cleared cache: {cache_dir}")
            except Exception as e:
                print(f"⚠️ Could not clear {cache_dir}: {e}")

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'clickhouse-connect',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def test_clickhouse_connection():
    """Test ClickHouse connection."""
    try:
        from clickhouse_connect import get_client
        
        client = get_client(
            host='clickhouse',
            username='default',
            password=''
        )
        
        # Test connection
        result = client.query("SELECT 1")
        print("✅ ClickHouse connection successful")
        return True
        
    except Exception as e:
        print(f"❌ ClickHouse connection failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure ClickHouse is running")
        print("2. Check if you're using Docker: docker-compose up -d")
        print("3. Verify ClickHouse host settings in config.py")
        return False

def run_dashboard():
    """Run the fixed dashboard."""
    print("\n🚀 Starting fixed dashboard...")
    
    # Use the fixed dashboard
    dashboard_file = "ui_dashboard_fixed.py"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file not found: {dashboard_file}")
        return False
    
    try:
        # Run with streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", dashboard_file, "--server.port", "8501"]
        print(f"Running: {' '.join(cmd)}")
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False

def main():
    """Main function to fix dashboard issues."""
    print("🔧 HFT Trading Dashboard Fix Tool")
    print("=" * 40)
    
    # Clear cache
    print("\n🧹 Clearing cache...")
    clear_streamlit_cache()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Test database connection
    print("\n🗄️ Testing database connection...")
    db_ok = test_clickhouse_connection()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first")
        return
    
    if not db_ok:
        print("\n❌ Database connection failed. Please fix database issues first")
        return
    
    print("\n✅ All checks passed!")
    print("\n🎯 Starting dashboard...")
    
    # Run the dashboard
    run_dashboard()

if __name__ == "__main__":
    main() 