#!/usr/bin/env python3
"""
Advanced Wikipedia XML dump extractor with automatic download and multiple export formats
Compatible with Python 3.13 - Uses mwxml library as an alternative to WikiExtractor

Features:
- Automatic download from Wikimedia dumps
- Export to JSONL, CSV, Parquet, and HuggingFace Dataset
- Progress tracking and logging
- Memory-efficient processing
"""

import mwxml
import bz2
import os
import sys
import re
import json
import csv
import logging
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Optional, Iterator

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    
try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wiki_extract.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WikiExtractor:
    """Enhanced Wikipedia dump extractor with multiple export formats"""
    
    def __init__(self, 
                 articles_per_file: int = 1000,
                 min_text_length: int = 100,
                 max_articles: Optional[int] = None):
        self.articles_per_file = articles_per_file
        self.min_text_length = min_text_length
        self.max_articles = max_articles
        self.base_url_template = "https://dumps.wikimedia.org/{lang}wiki/{date}/{lang}wiki-{date}-pages-articles-multistream.xml.bz2"
    
    def download_dump(self, lang: str = "th", date: str = "20250601", output_path: Optional[str] = None) -> str:
        """Download Wikipedia dump file"""
        if not output_path:
            output_path = f"{lang}wiki-{date}-pages-articles-multistream.xml.bz2"
        
        if os.path.exists(output_path):
            logger.info(f"File {output_path} already exists, skipping download")
            return output_path
            
        url = self.base_url_template.format(lang=lang, date=date)
        logger.info(f"Downloading {url}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)            
            print(f"\nDownload completed: {output_path}")
            logger.info(f"Successfully downloaded {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning specifically for Thai Wikipedia with comprehensive markup removal"""
        if not text:
            return ""
        
        # Remove wiki tables {| ... |}
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        
        # Remove HTML tables and cells
        text = re.sub(r'<table[^>]*>.*?</table>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<tr[^>]*>.*?</tr>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<td[^>]*>.*?</td>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<th[^>]*>.*?</th>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove wiki table markup
        text = re.sub(r'\{\|[^}]*\|\}', '', text, flags=re.DOTALL)
        text = re.sub(r'\|-[^\n]*', '', text)
        text = re.sub(r'\![^\n]*\|[^\n]*', '', text)
        text = re.sub(r'\|[^{}\n]*style="[^"]*"[^|\n]*', '', text)
        text = re.sub(r'\|\s*class="[^"]*"[^|\n]*', '', text)
        text = re.sub(r'\|\s*style="[^"]*"[^|\n]*', '', text)
        text = re.sub(r'\|[^|\n]*width="[^"]*"[^|\n]*', '', text)
        
        # Remove wiki templates {{...}} with better handling
        # Handle nested templates
        bracket_count = 0
        i = 0
        start_pos = -1
        while i < len(text):
            if i < len(text) - 1 and text[i:i+2] == '{{':
                if bracket_count == 0:
                    start_pos = i
                bracket_count += 2
                i += 2
            elif i < len(text) - 1 and text[i:i+2] == '}}':
                bracket_count -= 2
                if bracket_count == 0 and start_pos != -1:
                    text = text[:start_pos] + text[i+2:]
                    i = start_pos - 1
                    start_pos = -1
                else:
                    i += 2
            else:
                i += 1
        
        # Remove simple templates that might remain
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
        
        # Remove references and citations
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/?>', '', text)
        text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers [1], [2], etc.
        
        # Remove wiki links but keep text [[Link|Text]] -> Text, [[Link]] -> Link
        text = re.sub(r'\[\[([^|\]]*\|)?([^\]]*)\]\]', r'\2', text)
        
        # Remove external links [http://... text] -> text
        text = re.sub(r'\[https?://[^\s\]]+ ([^\]]*)\]', r'\1', text)
        text = re.sub(r'\[https?://[^\s\]]+\]', '', text)  # Remove bare URLs
        
        # Remove categories and files
        text = re.sub(r'\[\[(?:Category|หมวดหมู่):.*?\]\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\[(?:File|Image|ไฟล์|รูป):.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki formatting
        text = re.sub(r"'''([^']+)'''", r'\1', text)  # bold
        text = re.sub(r"''([^']+)''", r'\1', text)    # italic
        
        # Remove remaining wiki syntax
        text = re.sub(r'__[A-Z_]+__', '', text)  # __NOTOC__, __NOEDITSECTION__, etc.
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # HTML comments
        
        # Remove navigation and interface elements
        text = re.sub(r'thumb\|[^|]*\|', '', text)
        text = re.sub(r'left\|[^|]*\|', '', text)
        text = re.sub(r'right\|[^|]*\|', '', text)
        text = re.sub(r'\d+px\|', '', text)
        
        # Remove special characters and symbols that are not Thai text
        text = re.sub(r'[–—−]', '-', text)  # Normalize dashes
        text = re.sub(r'["""]', '"', text)  # Normalize quotes
        text = re.sub(r"[''']", "'", text)  # Normalize apostrophes
        
        # Remove lines that are mostly markup or numbers
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if len(line) < 5:  # Skip very short lines
                continue
            if re.match(r'^[^ก-๙]*$', line) and not re.search(r'[a-zA-Z]{3,}', line):
                # Skip lines with no Thai characters unless they have meaningful English
                continue
            if line.count('|') > 3 or line.count('{') > 2 or line.count('}') > 2:
                # Skip lines with too much markup
                continue
            if re.match(r'^[\s\-=\*\#\|\{\}]*$', line):
                # Skip lines with only markup characters
                continue
            clean_lines.append(line)
        
        text = '\n'.join(clean_lines)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n{3,}', '\n\n', text)  # More than 2 newlines to 2 newlines
        
        # Remove leading/trailing whitespace and clean up
        text = text.strip()
        
        # Final cleanup - remove remaining single characters and very short fragments
        lines = text.split('\n')
        final_lines = []
        for line in lines:
            line = line.strip()
            if len(line) >= 10:  # Only keep lines with substantial content
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def generate_wikipedia_url(self, title: str, lang: str = "th") -> str:
        """Generate Wikipedia URL for article"""
        # Replace spaces with underscores and encode
        title_encoded = title.replace(' ', '_')
        return f"https://{lang}.wikipedia.org/wiki/{title_encoded}"
    
    def extract_articles(self, dump_file: str, lang: str = "th") -> Iterator[Dict]:
        """Extract articles from Wikipedia dump as iterator"""
        logger.info(f"Processing {dump_file}...")
        
        # Open the bz2 compressed file
        if dump_file.endswith('.bz2'):
            file_obj = bz2.open(dump_file, 'rt', encoding='utf-8')
        else:
            file_obj = open(dump_file, 'r', encoding='utf-8')
        
        try:
            dump = mwxml.Dump.from_file(file_obj)
            
            article_count = 0
            
            for page in dump:
                # Skip redirects and non-main namespace pages
                if page.redirect or page.namespace != 0:
                    continue
                    
                # Get the latest revision
                latest_revision = None
                for revision in page:
                    latest_revision = revision
                    break
                
                if not latest_revision or not latest_revision.text:
                    continue
                
                # Clean the text
                cleaned_text = self.clean_text(latest_revision.text)
                
                # Skip articles that are too short after cleaning
                if len(cleaned_text) < self.min_text_length:
                    continue
                
                # Create article record
                article = {
                    "id": str(page.id),
                    "url": self.generate_wikipedia_url(page.title, lang),
                    "title": page.title,
                    "text": cleaned_text
                }
                
                yield article
                
                article_count += 1
                
                if article_count % 1000 == 0:
                    logger.info(f"Processed {article_count} articles...")
                
                # Stop if max_articles reached
                if self.max_articles and article_count >= self.max_articles:
                    break
                    
        finally:
            file_obj.close()
    
    def export_to_jsonl(self, articles: Iterator[Dict], output_path: str):
        """Export articles to JSONL format"""
        logger.info(f"Exporting to JSONL: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
    
    def export_to_csv(self, articles: Iterator[Dict], output_path: str):
        """Export articles to CSV format"""
        logger.info(f"Exporting to CSV: {output_path}")
        
        articles_list = list(articles)
        if not articles_list:
            logger.warning("No articles to export")
            return
            
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'url', 'title', 'text'])
            writer.writeheader()
            writer.writerows(articles_list)
    
    def export_to_parquet(self, articles: Iterator[Dict], output_path: str):
        """Export articles to Parquet format"""
        if not HAS_PANDAS:
            logger.error("Pandas not installed, cannot export to Parquet")
            return
            
        logger.info(f"Exporting to Parquet: {output_path}")
        
        articles_list = list(articles)
        if not articles_list:
            logger.warning("No articles to export")
            return
            
        df = pd.DataFrame(articles_list)
        df.to_parquet(output_path, index=False)
    
    def export_to_huggingface(self, articles: Iterator[Dict], output_path: str, push_to_hub: bool = False, repo_name: Optional[str] = None):
        """Export articles to HuggingFace Dataset format"""
        if not HAS_DATASETS:
            logger.error("HuggingFace Datasets not installed, cannot export to HF format")
            return
            
        logger.info(f"Exporting to HuggingFace Dataset: {output_path}")
        
        articles_list = list(articles)
        if not articles_list:
            logger.warning("No articles to export")
            return
            
        dataset = Dataset.from_list(articles_list)
        dataset.save_to_disk(output_path)
        
        if push_to_hub and repo_name:
            try:
                dataset.push_to_hub(repo_name)
                logger.info(f"Dataset pushed to HuggingFace Hub: {repo_name}")
            except Exception as e:
                logger.error(f"Failed to push to hub: {e}")
    
    def process_all_formats(self, 
                          dump_file: str, 
                          output_dir: str, 
                          lang: str = "th",
                          formats: List[str] = ["jsonl", "csv", "parquet", "hf"],
                          push_to_hub: bool = False,
                          repo_name: Optional[str] = None):
        """Process dump and export to all specified formats"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Base filename
        base_name = f"{lang}wiki_articles_{datetime.now().strftime('%Y%m%d')}"
        
        # Extract articles once and store in memory for multiple exports
        logger.info("Extracting all articles...")
        articles_list = list(self.extract_articles(dump_file, lang))
        logger.info(f"Extracted {len(articles_list)} articles")
        
        # Export to each format
        for fmt in formats:
            if fmt == "jsonl":
                output_path = os.path.join(output_dir, f"{base_name}.jsonl")
                self.export_to_jsonl(iter(articles_list), output_path)
                
            elif fmt == "csv":
                output_path = os.path.join(output_dir, f"{base_name}.csv")
                self.export_to_csv(iter(articles_list), output_path)
                
            elif fmt == "parquet":
                output_path = os.path.join(output_dir, f"{base_name}.parquet")
                self.export_to_parquet(iter(articles_list), output_path)
                
            elif fmt == "hf":
                output_path = os.path.join(output_dir, f"{base_name}_dataset")
                self.export_to_huggingface(iter(articles_list), output_path, push_to_hub, repo_name)
        
        logger.info(f"Processing complete! All files saved to {output_dir}")


def main():
    """Main function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Wikipedia Dump Extractor')
    parser.add_argument('--dump-file', type=str, help='Path to Wikipedia dump file')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--lang', type=str, default='th', help='Wikipedia language code')
    parser.add_argument('--date', type=str, default='20250601', help='Dump date (YYYYMMDD)')
    parser.add_argument('--download', action='store_true', help='Download dump file automatically')
    parser.add_argument('--formats', nargs='+', default=['jsonl', 'csv', 'parquet', 'hf'], 
                       choices=['jsonl', 'csv', 'parquet', 'hf'], help='Export formats')
    parser.add_argument('--articles-per-file', type=int, default=1000, help='Articles per file')
    parser.add_argument('--min-text-length', type=int, default=100, help='Minimum text length')
    parser.add_argument('--max-articles', type=int, help='Maximum articles to process')
    parser.add_argument('--push-to-hub', action='store_true', help='Push to HuggingFace Hub')
    parser.add_argument('--repo-name', type=str, help='HuggingFace repository name')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = WikiExtractor(
        articles_per_file=args.articles_per_file,
        min_text_length=args.min_text_length,
        max_articles=args.max_articles
    )
    
    # Download dump file if requested
    if args.download:
        dump_file = extractor.download_dump(args.lang, args.date)
    else:
        dump_file = args.dump_file
        if not dump_file or not os.path.exists(dump_file):
            logger.error("Dump file not found. Use --download or provide valid --dump-file")
            sys.exit(1)
    
    # Process all formats
    extractor.process_all_formats(
        dump_file=dump_file,
        output_dir=args.output_dir,
        lang=args.lang,
        formats=args.formats,
        push_to_hub=args.push_to_hub,
        repo_name=args.repo_name
    )


if __name__ == "__main__":
    main()
