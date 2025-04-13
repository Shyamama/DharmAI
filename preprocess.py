#!/usr/bin/env python3
from datasets import Dataset, concatenate_datasets
import glob
import re
from indic_transliteration import sanscript  # For Sanskrit transliteration (optional)
from pathlib import Path

def clean_text(text):
    """Remove verse numbers, footnotes, and normalize whitespace"""
    # Remove verse numbers (e.g., "1.2.3" or "[v10]")
    text = re.sub(r'(\[v\d+\]|\d+\.\d+\.\d+)', '', text)
    
    # Remove publisher annotations (e.g., "(*comment*)")
    text = re.sub(r'\(\*.*?\*\)', '', text)
    
    # Standardize whitespace
    text = ' '.join(text.split())
    return text.strip()

def transliterate_to_IAST(text):
    """Convert Sanskrit text to IAST (optional)"""
    try:
        return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
    except:
        return text  # Fallback if non-Sanskrit text

def load_and_process_text(filepath):
    """Load a scripture file and clean its contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [clean_text(line) for line in f if line.strip()]
        if "upanishad" in filepath.lower():  # Apply transliteration only to Sanskrit texts
            lines = [transliterate_to_IAST(line) for line in lines]
        return Dataset.from_dict({"text": lines})

def main():
    # Create output directory
    Path("data/pretrain_dataset").mkdir(parents=True, exist_ok=True)
    
    # Load all scripture files
    datasets = []
    for filepath in glob.glob("data/raw_scripts/*.txt"):
        print(f"Processing {filepath}...")
        ds = load_and_process_text(filepath)
        datasets.append(ds)
        print(f"  → Loaded {len(ds)} verses")

    # Combine datasets
    combined_dataset = concatenate_datasets(datasets)
    print(f"\nTotal verses: {len(combined_dataset)}")
    
    # Save for training
    combined_dataset.save_to_disk("data/pretrain_dataset")
    print("Saved preprocessed dataset to data/pretrain_dataset")

if __name__ == "__main__":
    main()