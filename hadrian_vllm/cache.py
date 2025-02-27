# %%
# cache.py
import os
import json
import threading


class PersistentCache:
    """A thread-safe dictionary-like object that persists to disk."""

    def __init__(self, cache_file_path="data/cache.json"):
        self.cache_file_path = cache_file_path
        self.lock = threading.Lock()
        self._ensure_cache_file_exists()
        self._cache = self._load_cache()

    def _ensure_cache_file_exists(self):
        """Ensure the cache file exists"""
        os.makedirs(os.path.dirname(self.cache_file_path), exist_ok=True)
        if not os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, "w") as f:
                json.dump({}, f)

    def _load_cache(self):
        """Load the cache from disk"""
        try:
            with open(self.cache_file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_cache(self):
        """Save the cache to disk"""
        with open(self.cache_file_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def __getitem__(self, key):
        """Get an item from the cache"""
        with self.lock:
            return self._cache.get(key)

    def __setitem__(self, key, value):
        """Set an item in the cache and persist it to disk"""
        if value:
            with self.lock:
                self._cache[key] = value
                self._save_cache()
        else:
            print("WARN: Not caching `{value}`")

    def __contains__(self, key):
        """Check if key exists in cache"""
        with self.lock:
            return key in self._cache

    def get(self, key, default=None):
        """Get an item with a default value if not found"""
        with self.lock:
            return self._cache.get(key, default)

    def clear(self):
        """Clear the cache"""
        with self.lock:
            self._cache = {}
            self._save_cache()


if __name__ == "__main__":
    c = PersistentCache()
    print(c._cache)
