#!/usr/bin/env python3
"""
Configuration and setup script for WikiDumper
"""

import os
import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("🔧 Installing dependencies...")
    
    # Install core dependencies
    core_deps = ["mwxml", "requests"]
    for dep in core_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {dep}: {e}")
    
    # Install optional dependencies
    optional_deps = ["pandas", "pyarrow", "datasets"]
    for dep in optional_deps:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except Exception as e:
            print(f"⚠️  Optional dependency {dep} failed to install: {e}")
            print(f"   You can install it manually: pip install {dep}")

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    dirs = ["output", "downloads", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def create_example_scripts():
    """Create example usage scripts"""
    print("📝 Creating example scripts...")
    
    # Example 1: Simple extraction
    example1 = '''#!/usr/bin/env python3
"""
Example 1: Simple Wikipedia extraction
"""

from extract_wiki import WikiExtractor

def main():
    extractor = WikiExtractor(max_articles=1000)  # Limit for testing
    
    # Download and process Thai Wikipedia
    dump_file = extractor.download_dump(lang="th", date="20250601")
    
    # Export to all formats
    extractor.process_all_formats(
        dump_file=dump_file,
        output_dir="output/thai_sample",
        lang="th",
        formats=["jsonl", "csv"]
    )

if __name__ == "__main__":
    main()
'''
    
    with open("example_simple.py", "w", encoding="utf-8") as f:
        f.write(example1)
    
    # Example 2: Advanced with HuggingFace
    example2 = '''#!/usr/bin/env python3
"""
Example 2: Advanced extraction with HuggingFace integration
"""

from extract_wiki import WikiExtractor

def main():
    extractor = WikiExtractor(
        articles_per_file=5000,
        min_text_length=200,
        max_articles=10000  # Adjust as needed
    )
    
    # Process multiple languages
    languages = ["th", "en", "ja"]
    
    for lang in languages:
        print(f"Processing {lang} Wikipedia...")
        
        # Download dump
        dump_file = extractor.download_dump(lang=lang, date="20250601")
        
        # Export to all formats including HuggingFace
        extractor.process_all_formats(
            dump_file=dump_file,
            output_dir=f"output/{lang}_wikipedia",
            lang=lang,
            formats=["jsonl", "parquet", "hf"],
            push_to_hub=False,  # Set to True if you want to push to HF Hub
            repo_name=f"your-username/{lang}-wikipedia-articles"
        )

if __name__ == "__main__":
    main()
'''
    
    with open("example_advanced.py", "w", encoding="utf-8") as f:
        f.write(example2)
    
    print("✅ Created example scripts: example_simple.py, example_advanced.py")

def main():
    """Main setup function"""
    print("🚀 WikiDumper Setup")
    print("=" * 50)
    
    install_dependencies()
    setup_directories()
    create_example_scripts()
    
    print("\n✨ Setup complete!")
    print("\n📖 Quick start:")
    print("1. Run: python example_simple.py")
    print("2. Or use CLI: python extract_wiki.py --download --lang th --formats jsonl csv")
    print("\n📚 For more options: python extract_wiki.py --help")

if __name__ == "__main__":
    main()
