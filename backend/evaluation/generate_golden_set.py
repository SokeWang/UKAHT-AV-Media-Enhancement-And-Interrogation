import os
import json
from psycopg2.extras import RealDictCursor
from backend.db.database import get_connection

def generate_golden_set():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    output_path = os.path.join(project_root, "golden_test_set.json")
    
    print("Connecting to PostgreSQL database...")
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, title, category, description, base_code, subject_type 
                FROM assets 
                WHERE (description IS NOT NULL AND description != '')
                   OR (title IS NOT NULL AND title != '')
            """)
            rows = cursor.fetchall()

    print(f"Found {len(rows)} assets in PostgreSQL database.")
    
    query_to_assets = {}
    for row in rows:
        asset_id = row["id"]
        # Use description if available, else title or clean metadata string
        desc = (row.get("description") or "").strip()
        title = (row.get("title") or "").strip()
        
        query_text = desc if desc else title
        if not query_text:
            category = (row.get("category") or "").strip()
            base_code = (row.get("base_code") or "").strip()
            if category or base_code:
                query_text = f"{category} photo at Base {base_code}".strip()
                
        if not query_text:
            continue

        query_text = " ".join(query_text.split())
        if query_text not in query_to_assets:
            query_to_assets[query_text] = []
        query_to_assets[query_text].append(asset_id)
        
    golden_entries = []
    # Deduplicate queries and take representative entries
    sorted_queries = sorted(query_to_assets.items(), key=lambda x: len(x[1]), reverse=True)[:50]
    
    for query, asset_ids in sorted_queries:
        golden_entries.append({
            "asset_id": asset_ids[0],
            "caption": query,
            "query": query,
            "relevant_ids": asset_ids
        })
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_entries, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully generated and wrote {len(golden_entries)} golden benchmark query pairs to '{output_path}'.")

if __name__ == "__main__":
    generate_golden_set()

