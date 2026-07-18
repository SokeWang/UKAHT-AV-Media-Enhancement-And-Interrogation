import os
import json
import sqlite3

def generate_golden_set():
    # Locate files relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    db_path = os.path.join(project_root, "backend", "db.sqlite")
    output_path = os.path.join(project_root, "golden_test_set.json")
    
    if not os.path.exists(db_path):
        print(f"[ERROR] SQLite database not found at '{db_path}'. Cannot generate golden set.")
        return
        
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query assets with usable metadata
    cursor.execute("""
        SELECT id, base_code, subject_type, category, title 
        FROM assets 
        WHERE (base_code IS NOT NULL AND base_code != '')
           OR (subject_type IS NOT NULL AND subject_type != '')
           OR (category IS NOT NULL AND category != '')
    """)
    rows = cursor.fetchall()
    print(f"Found {len(rows)} assets with metadata candidates.")
    
    # Group assets by generated query
    query_to_assets = {}
    
    for row in rows:
        asset_id, base_code, subject_type, category, title = row
        
        # Build query candidate
        parts = []
        if subject_type:
            parts.append(subject_type.lower())
        else:
            parts.append("archival content")
            
        if base_code:
            parts.append(f"at base {base_code.replace('Base ', '')}")
            
        if category:
            parts.append(f"focusing on {category.lower()}")
            
        query_text = " ".join(parts).strip()
        # Clean double spaces or weird formats
        query_text = " ".join(query_text.split())
        
        if not query_text:
            continue
            
        if query_text not in query_to_assets:
            query_to_assets[query_text] = []
        query_to_assets[query_text].append(asset_id)
        
    # Build list of golden entries
    golden_entries = []
    # Cap total entries to around 60 representative queries to keep evaluation swift yet rigorous
    sorted_queries = sorted(query_to_assets.items(), key=lambda x: len(x[1]), reverse=True)[:60]
    
    for query, asset_ids in sorted_queries:
        golden_entries.append({
            "asset_id": asset_ids[0], # primary anchor asset
            "caption": query,
            "query": query,
            "relevant_ids": asset_ids
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_entries, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated and wrote {len(golden_entries)} golden benchmark query pairs to '{output_path}'.")

if __name__ == "__main__":
    generate_golden_set()
