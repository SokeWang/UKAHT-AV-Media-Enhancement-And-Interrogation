import numpy as np
import os
from backend.db.database import get_connection
from psycopg2.extras import RealDictCursor

def clean_duplicates():
    try:
        conn = get_connection()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    with conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get all assets
            cursor.execute(
                "SELECT id, url, title, embedding FROM assets"
            )
            rows = cursor.fetchall()

            print(f"Total assets in database: {len(rows)}")

            seen_embeddings = []
            to_delete = []

            for row in rows:
                asset_id = row["id"]
                title = row["title"]
                emb_bytes = row["embedding"]

                if not emb_bytes:
                    continue

                new_emb = np.frombuffer(bytes(emb_bytes), dtype=np.float32)
                norm = np.linalg.norm(new_emb)
                if norm > 0:
                    new_emb = new_emb / norm

                # Compare with already seen embeddings
                is_duplicate = False
                for seen_id, seen_title, seen_emb in seen_embeddings:
                    similarity = np.dot(new_emb, seen_emb)
                    if similarity >= 0.99:
                        print(f"Found duplicate: '{title}' ({asset_id}) is a visual duplicate of '{seen_title}' ({seen_id}) [Similarity: {similarity:.4f}]")
                        is_duplicate = True
                        to_delete.append(asset_id)
                        break

                if not is_duplicate:
                    seen_embeddings.append((asset_id, title, new_emb))

            if to_delete:
                print(f"\nDeleting {len(to_delete)} duplicate assets from the database...")
                cursor.execute(
                    "DELETE FROM assets WHERE id = ANY(%s)", 
                    (to_delete,)
                )
                print("Cleanup completed successfully!")
            else:
                print("\nNo duplicates found in the database.")
    
    conn.close()

if __name__ == "__main__":
    clean_duplicates()
