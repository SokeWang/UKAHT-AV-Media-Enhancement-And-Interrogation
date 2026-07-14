import sqlite3
import numpy as np
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db.sqlite")

def clean_duplicates():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all assets
    rows = cursor.execute(
        "SELECT id, url, title, embedding FROM assets"
    ).fetchall()

    print(f"Total assets in database: {len(rows)}")

    seen_embeddings = []
    to_delete = []

    for row in rows:
        asset_id = row["id"]
        title = row["title"]
        emb_bytes = row["embedding"]

        if not emb_bytes:
            continue

        new_emb = np.frombuffer(emb_bytes, dtype=np.float32)
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
        cursor.execute(f"DELETE FROM assets WHERE id IN ({','.join(['?']*len(to_delete))})", to_delete)
        conn.commit()
        print("Cleanup completed successfully!")
    else:
        print("\nNo duplicates found in the database.")

    conn.close()

if __name__ == "__main__":
    clean_duplicates()
