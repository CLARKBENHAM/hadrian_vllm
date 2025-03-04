# %%
# # cache.py
import os
import json
import threading


class PersistentCache:
    """A thread-safe dictionary-like object that persists to disk incrementally using JSON Lines."""

    def __init__(self, cache_file_path="data/cache.jsonl"):
        self.cache_file_path = cache_file_path
        self.lock = threading.Lock()
        self._ensure_cache_file_exists()
        self._cache = self._load_cache()
        self._store_count = len(self._cache)

    def _ensure_cache_file_exists(self):
        """Ensure the cache file exists"""
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        if not os.path.exists(self.cache_file_path):
            # Create an empty file
            open(self.cache_file_path, "a").close()

    def _load_cache(self):
        """Load the cache from disk (each line is a JSON object with a single key-value pair)"""
        cache = {}
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            cache.update(data)
                        except json.JSONDecodeError:
                            continue
        return cache

    def _save_cache(self):
        """Write the current cache to disk as JSON Lines (used during compaction)."""
        with open(self.cache_file_path, "w") as f:
            for key, value in self._cache.items():
                f.write(json.dumps({key: value}) + "\n")

    def _remove_duplicates(self):
        """
        Remove duplicate entries from the cache file by rewriting the file
        with only the current key-value pairs.
        """
        self._save_cache()

    def __getitem__(self, key):
        """Get an item from the cache"""
        with self.lock:
            return self._cache.get(key)

    def __setitem__(self, key, value):
        """Set an item in the cache and append only the updated item to disk"""
        with self.lock:
            if value == self.get(key):
                return
            self._cache[key] = value
            with open(self.cache_file_path, "a") as f:
                f.write(json.dumps({key: value}) + "\n")
            self._store_count += 1
            if self._store_count % 1000 == 0:
                self._remove_duplicates()

    def __contains__(self, key):
        """Check if key exists in cache"""
        with self.lock:
            return key in self._cache

    def get(self, key, default=None):
        """Get an item with a default value if not found"""
        with self.lock:
            return self._cache.get(key, default)

    def clear(self):
        """Clear the cache both in memory and on disk"""
        with self.lock:
            self._cache = {}
            # Overwrite the file to clear it
            with open(self.cache_file_path, "w") as f:
                f.write("")


def _migrate_cache(old_cache_path="data/old_cache.json", new_cache_path="data/cache.jsonl"):
    """
    Migrate the old JSON cache to the new JSON Lines (JSONL) format.

    Reads the entire old cache (a single JSON object) from old_cache_path and writes
    each key–value pair as its own JSON object (one per line) to new_cache_path.

    Args:
        old_cache_path (str): Path to the old cache JSON file.
        new_cache_path (str): Path to the new JSONL cache file.
    """
    # Ensure the directory for the new cache file exists
    os.makedirs(os.path.dirname(new_cache_path), exist_ok=True)

    # Load the old cache data
    try:
        with open(old_cache_path, "r") as f:
            old_data = json.load(f)
    except Exception as e:
        print(f"Error reading old cache: {e}")
        return

    # Write each key-value pair as a separate JSON object (one per line)
    try:
        with open(new_cache_path, "w") as f:
            for key, value in old_data.items():
                # Each line is a JSON object with a single key-value pair.
                json_line = json.dumps({key: value})
                f.write(json_line + "\n")
        print(f"Migration complete: {len(old_data)} entries migrated to {new_cache_path}")
    except Exception as e:
        print(f"Error writing new cache: {e}")


if __name__ == "__main__":
    _migrate_cache()
    c = PersistentCache()
    print(c._cache)
