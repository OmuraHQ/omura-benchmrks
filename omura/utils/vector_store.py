"""Vector store for storing and searching embeddings.

Supports multiple backends:
- FAISS (CPU): Reliable, exact or ANN search. Default.
- cuVS (GPU): High-performance ANN search on NVIDIA GPUs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from omura.utils.blob_catalog import CATALOG_DB_PATH, init_catalog_db

# Try importing backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    import cudf
    import cupy as cp
    from cuvs.neighbors import cagra
    CUVS_AVAILABLE = True
except ImportError:
    cudf = None
    cp = None
    cagra = None
    CUVS_AVAILABLE = False


# Vector store directory (legacy default)
VECTOR_STORE_DIR = Path(os.getenv("OMURA_VECTOR_STORE_DIR", "data/vector_index"))
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Legacy embedding output dimension (default 768)
EMBEDDING_DIM = int(os.getenv("OMURA_EMBEDDING_DIM", "768"))


class VectorStore:
    """Vector store supporting both cuVS (GPU) and FAISS (CPU/GPU) backends.
    
    Backend is selected via OMURA_VECTOR_BACKEND environment variable:
    - 'faiss': Use FAISS (CPU by default, robust). Default.
    - 'cuvs': Use NVIDIA cuVS (GPU, experimental/fast).
    """

    def __init__(self, index_path: Optional[Path] = None, embedding_dim: Optional[int] = None):
        """Initialize vector store.

        Args:
            index_path: Path to save/load the index. If None, uses default.
            embedding_dim: Per-instance embedding dimension. If None, uses the
                module-level EMBEDDING_DIM. Lets a separate store (e.g. a 512-d
                CLAP audio index) coexist with the main store in one process.
        """
        self.embedding_dim = int(embedding_dim) if embedding_dim else EMBEDDING_DIM
        self.backend = os.getenv("OMURA_VECTOR_BACKEND", "faiss").lower()
        
        # Validate backend availability
        if self.backend == "cuvs":
            if not CUVS_AVAILABLE:
                print("[VectorStore] Warning: cuVS requested but not available. Falling back to FAISS.")
                self.backend = "faiss"
        
        if self.backend == "faiss":
            if not FAISS_AVAILABLE:
                # If neither is available, we can't do anything
                if CUVS_AVAILABLE:
                    print("[VectorStore] Warning: FAISS not available. Falling back to cuVS.")
                    self.backend = "cuvs"
                else:
                    raise ImportError("No vector backend available. Install `faiss-cpu` or `cuvs-cu13`.")
            else:
                print("[VectorStore] Using FAISS backend")
        else:
            print("[VectorStore] Using cuVS backend")

        # cuVS GPU configuration
        self.cuvs_gpu_id = -1
        if self.backend == "cuvs":
            cuvs_gpu_id_env = os.getenv("OMURA_CUVS_GPU_ID", "1")
            try:
                self.cuvs_gpu_id = int(cuvs_gpu_id_env)
            except ValueError:
                self.cuvs_gpu_id = 1
            
            # Validate GPU
            if cp is not None and cp.cuda.is_available():
                num_gpus = cp.cuda.runtime.getDeviceCount()
                if self.cuvs_gpu_id < 0 or self.cuvs_gpu_id >= num_gpus:
                    print(f"[VectorStore] Warning: Invalid cuVS GPU ID {self.cuvs_gpu_id}, using GPU 1")
                    self.cuvs_gpu_id = 1
                try:
                    cp.cuda.Device(self.cuvs_gpu_id).use()
                    print(f"[VectorStore] Using GPU {self.cuvs_gpu_id} for cuVS operations")
                except Exception as e:
                    print(f"[VectorStore] Warning: Failed to set cuVS GPU: {e}")
            else:
                self.cuvs_gpu_id = -1
                print("[VectorStore] CUDA not available for cuVS, operations may fail")

        # Paths
        self.index_path = index_path or VECTOR_STORE_DIR / f"vector_index.{'cagra' if self.backend == 'cuvs' else 'faiss'}"
        self.metadata_path = self.index_path.parent / "metadata.json"
        self.cursor_path = self.index_path.parent / "cursor.json"
        self.backup_dir = self.index_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_db_path = Path(
            os.getenv("OMURA_CATALOG_DB_PATH", str(CATALOG_DB_PATH))
        )
        init_catalog_db(self.catalog_db_path)
        
        # Storage for embeddings (in-memory)
        # Mapping: blob_id -> embedding vector
        self.embeddings_dict: Dict[str, np.ndarray] = {}
        # Ordered list for building index (maintains order)
        self.embeddings_list: List[np.ndarray] = []
        # Mapping: position in embeddings_list -> blob_id
        self.position_to_blob_id: List[str] = []
        
        # Metadata storage: blob_id -> JSON string (nested JSON as string)
        self.metadata: Dict[str, str] = {}
        
        # Cursor for incremental indexing
        self.cursor: Dict = {
            "last_page": 0,
            "last_processed_blob_id": None,
            "last_updated": None,
        }
        
        # Index instance (either cuVS CAGRA or FAISS Index)
        self.index = None
        self._is_built = False
        
        # GPU embeddings cache for cuVS fallback
        self.gpu_embeddings = None
        
        # Thread locks for thread-safe operations
        # `_lock` is kept for compatibility with older callers that expect a
        # coarse-grained store lock (e.g. cluster rebuild snapshots).
        self._lock = threading.RLock()
        self._build_lock = threading.Lock()  # For index building
        self._save_lock = threading.Lock()  # For saving (prevent concurrent saves)
        self._search_lock = threading.Lock()  # For search (prevent conflicts with other GPU ops)

        # Initialize empty index for FAISS immediately
        if self.backend == "faiss":
            self._init_faiss_index()

    def _init_faiss_index(self):
        """Initialize a new FAISS index."""
        # Use IndexFlatIP (Inner Product) for exact search with cosine similarity on normalized vectors
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self._is_built = True

    def add(
        self,
        embedding: np.ndarray,
        blob_id: str,
        mime_type: str,
        size: int,
        **extra_metadata,
    ) -> None:
        """Add an embedding with metadata.

        Args:
            embedding: Normalized embedding vector (shape: [EMBEDDING_DIM])
            blob_id: Walrus blob ID (PRIMARY KEY)
            mime_type: Detected MIME type
            size: Blob size in bytes
            **extra_metadata: Additional metadata fields
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"got {embedding.shape[0]}"
            )

        # Check if blob_id already exists
        is_update = blob_id in self.embeddings_dict
        
        # Store metadata as nested JSON string
        metadata_dict = {
            "size": size,
            "owner": extra_metadata.get("owner"),
            "expiresAt": extra_metadata.get("end_epoch"),  # expiresAt
            "is_nsfw": extra_metadata.get("is_nsfw", False),
            "nsfw_score": extra_metadata.get("nsfw_score"),
            "mime_type": mime_type,
            "extension": extra_metadata.get("extension"),
            "kind": extra_metadata.get("kind"),
            "is_quilt": bool(extra_metadata.get("is_quilt", False)),
            "parent_quilt_id": extra_metadata.get("parent_quilt_id"),
            "quilt_identifier": extra_metadata.get("quilt_identifier"),
            "quilt_tags_json": extra_metadata.get("quilt_tags_json"),
        }
        self.metadata[blob_id] = json.dumps(metadata_dict)
        self._sync_blob_metadata_sqlite(
            blob_id=blob_id,
            mime_type=mime_type,
            size=size,
            metadata_dict=metadata_dict,
        )

        # Update in-memory storage
        if not is_update:
            self.embeddings_dict[blob_id] = embedding
            self.embeddings_list.append(embedding)
            self.position_to_blob_id.append(blob_id)
            
            # For FAISS, we can add incrementally
            if self.backend == "faiss" and self.index is not None:
                with self._build_lock:
                    vector = embedding.reshape(1, -1).astype('float32')
                    self.index.add(vector)
        else:
            # Update existing
            if blob_id in self.position_to_blob_id:
                pos = self.position_to_blob_id.index(blob_id)
                self.embeddings_dict[blob_id] = embedding
                self.embeddings_list[pos] = embedding

                # FAISS Flat doesn't support easy updates without IDMap or rebuilding
                # For now, we accept that the old vector remains in FAISS until rebuild
                # or we assume append-only mostly.
                # Ideally, we should set _is_built = False to trigger rebuild later
                if self.backend == "faiss":
                    # Simple strategy: if update, mark for rebuild
                    self._is_built = False
                    print(f"[VectorStore] Blob {blob_id} updated, marking FAISS index for rebuild.")
                elif self.backend == "cuvs":
                    self._is_built = False
                    print(f"[VectorStore] Blob {blob_id} updated, marking cuVS index for rebuild.")
            else:
                # Should not happen
                self.embeddings_dict[blob_id] = embedding
                self.embeddings_list.append(embedding)
                self.position_to_blob_id.append(blob_id)
                if self.backend == "faiss" and self.index is not None:
                    with self._build_lock:
                        vector = embedding.reshape(1, -1).astype('float32')
                        self.index.add(vector)
        
        # For cuVS, we always need to rebuild/add in batch usually, 
        # but here we just mark as dirty if using cuVS or if update happened
        if self.backend == "cuvs" or is_update:
            self._is_built = False

    def build_index(self, graph_degree: int = 16, intermediate_graph_degree: int = 64) -> None:
        """Build the index for fast similarity search.

        Args:
            graph_degree: Degree of the final graph (cuVS only)
            intermediate_graph_degree: Degree of the intermediate graph (cuVS only)
        """
        # Thread-safe: prevent concurrent index building
        if not self._build_lock.acquire(blocking=False):
            print("[VectorStore] Index build already in progress, skipping...")
            return
        
        try:
            if not self.embeddings_list:
                return

            # Skip if already built and fresh
            if self._is_built and self.index is not None:
                # For FAISS Flat, if we added incrementally, we might be good.
                # But if we marked False (due to update), we proceed.
                if self.backend == "faiss":
                    # Check if counts match
                    if self.index.ntotal == len(self.embeddings_list):
                        return

            # Convert embeddings to numpy float32
            embeddings_array = np.array(self.embeddings_list, dtype=np.float32)
            if embeddings_array.size == 0:
                return
            
            num_embeddings = len(self.embeddings_list)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if self.backend == "faiss":
                print(f"[VectorStore] Building FAISS index from {num_embeddings} embeddings...")
                try:
                    # Re-initialize index
                    self._init_faiss_index()
                    self.index.add(embeddings_array)
                    self._is_built = True
                    print("[VectorStore] FAISS index built successfully")
                except Exception as e:
                    print(f"[VectorStore] Error building FAISS index: {e}")
                    import traceback
                    traceback.print_exc()
                    self._is_built = False

            elif self.backend == "cuvs":
                print(f"[VectorStore] Building cuVS CAGRA index from {num_embeddings} embeddings...")
                
                # Clear old index
                if self.index is not None:
                    try:
                        del self.index
                        self.index = None
                    except Exception:
                        pass

                try:
                    # Set cuVS GPU device and clear cache
                    if self.cuvs_gpu_id != -1:
                        try:
                            with cp.cuda.Device(self.cuvs_gpu_id):
                                # Clear GPU cache
                                cp.get_default_memory_pool().free_all_blocks()
                                cp.get_default_pinned_memory_pool().free_all_blocks()
                                
                                # Convert embeddings to GPU array on cuVS GPU
                                gpu_embeddings = cp.asarray(embeddings_array, order='C', dtype=np.float32)
                                if not gpu_embeddings.flags.c_contiguous:
                                    gpu_embeddings = cp.ascontiguousarray(gpu_embeddings)
                                
                                # Store for fallback
                                self.gpu_embeddings = gpu_embeddings
                        except Exception as gpu_error:
                            print(f"[VectorStore] Warning: Error setting cuVS GPU device: {gpu_error}")
                            gpu_embeddings = cp.asarray(embeddings_array)
                    else:
                        # Fallback (won't work well for cuVS but avoids crash before call)
                        gpu_embeddings = cp.asarray(embeddings_array, order='C', dtype=np.float32)

                    # Build CAGRA index
                    index_params = cagra.IndexParams(metric="cosine")
                    self.index = cagra.build(index_params, gpu_embeddings)
                    self._is_built = True
                    print("[VectorStore] cuVS index built successfully")
                except Exception as e:
                    print(f"[VectorStore] Error building cuVS index: {e}")
                    import traceback
                    traceback.print_exc()
                    self.index = None
                    self._is_built = False
        finally:
            self._build_lock.release()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        kind_filter: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Normalized query embedding (shape: [EMBEDDING_DIM])
            top_k: Number of results to return
            kind_filter: If set (e.g. 'image', 'video', 'audio'), restrict results to that kind.

        Returns:
            List of (metadata_dict, similarity_score) tuples, sorted by similarity
        """
        count = self.size()
        if count == 0:
            return []

        # Automatically build index if needed
        if not self._is_built:
            if not self.embeddings_list:
                return []
            self.build_index()

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # When filtering by kind, fetch more candidates then filter down
        k = min(top_k * 10, count) if kind_filter else min(top_k, count)

        indices_cpu = []
        distances_cpu = []

        if self.backend == "faiss":
            if self.index is None:
                # Should not happen if build_index succeeded, but safety check
                return []
            
            # FAISS expects float32
            query_array = query_embedding.reshape(1, -1).astype(np.float32)
            
            try:
                # Search FAISS index
                distances, faiss_indices = self.index.search(query_array, k)

                # faiss_indices[0] are indices, distances[0] are scores (similarities for InnerProduct)
                indices_cpu = faiss_indices[0]
                distances_cpu = distances[0]
                
            except Exception as e:
                print(f"[VectorStore] FAISS search failed: {e}")
                return []

        elif self.backend == "cuvs":
            if self.cuvs_gpu_id == -1:
                raise RuntimeError("CUDA not available for cuVS operations")
            
            with self._search_lock:
                with cp.cuda.Device(self.cuvs_gpu_id):
                    query_array = np.ascontiguousarray(
                        query_embedding.reshape(1, -1), dtype=np.float32
                    )
                    query_gpu = cp.asarray(query_array, order='C', dtype=np.float32)
                    if not query_gpu.flags.c_contiguous:
                        query_gpu = cp.ascontiguousarray(query_gpu)

                    search_params = cagra.SearchParams()
                    
                    try:
                        distances, indices = cagra.search(
                            search_params, self.index, query_gpu, k=k
                        )
                    except Exception as e:
                        print(f"[VectorStore] Warning: CAGRA search failed: {e}")
                        print("[VectorStore] Falling back to exact search (brute force)...")
                        
                        try:
                            # Fallback: Brute force on GPU
                            if not hasattr(self, 'gpu_embeddings') or self.gpu_embeddings is None:
                                 embeddings_array = np.ascontiguousarray(
                                    np.array(self.embeddings_list, dtype=np.float32)
                                 )
                                 self.gpu_embeddings = cp.asarray(embeddings_array, order='C', dtype=np.float32)

                            # Cosine similarity
                            similarities = cp.dot(query_gpu, self.gpu_embeddings.T)
                            top_k_indices = cp.argsort(similarities[0])[::-1][:k]
                            top_k_sims = similarities[0][top_k_indices]
                            
                            indices = top_k_indices.reshape(1, -1)
                            distances = (1.0 - top_k_sims).reshape(1, -1) # convert back to "distance" format for unified processing
                            
                        except Exception as fallback_error:
                            print(f"[VectorStore] Brute force fallback failed: {fallback_error}")
                            if "type mismatch" in str(e) or "raft" in str(e).lower():
                                # Force rebuild index next time
                                self._is_built = False
                                self.index = None
                            raise

                    # Process results from GPU
                    try:
                        distances_np = cp.asnumpy(distances)
                        indices_np = cp.asnumpy(indices)
                        
                        if distances_np.ndim == 2:
                            distances_cpu_raw = distances_np[0]
                            indices_cpu = indices_np[0]
                        else:
                            distances_cpu_raw = distances_np
                            indices_cpu = indices_np
                        
                        # Convert CAGRA distance to similarity
                        # Assuming cosine metric: sim = 1.0 - dist
                        distances_cpu = [1.0 - float(d) for d in distances_cpu_raw]

                    except Exception as e:
                        print(f"[VectorStore] Error processing search results: {e}")
                        return []

        # Map results to metadata
        results = []
        for idx, score in zip(indices_cpu, distances_cpu):
            idx_int = int(idx)
            if idx_int == -1:
                continue
            if idx_int >= len(self.position_to_blob_id):
                continue
            blob_id = self.position_to_blob_id[idx_int]
            metadata = self._get_metadata_by_blob_id(blob_id)
            if not metadata:
                continue
            if kind_filter and metadata.get("kind") != kind_filter:
                continue
            results.append((metadata, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _get_metadata_by_blob_id(self, blob_id: str) -> Optional[Dict]:
        """Get blob metadata by blob_id from JSON storage."""
        if blob_id not in self.metadata:
            return None
        try:
            raw = self.metadata[blob_id]
            if isinstance(raw, dict):
                meta = raw.copy()
            else:
                meta = json.loads(raw)
                # Handle legacy double-encoded metadata values.
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if not isinstance(meta, dict):
                    return None
            meta["blob_id"] = blob_id
            return meta
        except (json.JSONDecodeError, TypeError):
            return None

    def _sync_blob_metadata_sqlite(
        self, *, blob_id: str, mime_type: str, size: int, metadata_dict: Dict
    ) -> None:
        """Persist metadata into blob_catalog.sqlite so queue+metadata stay durable."""
        now = datetime.utcnow().isoformat()
        try:
            with sqlite3.connect(str(self.catalog_db_path), timeout=30) as conn:
                conn.execute(
                    """
                    INSERT INTO blobs (
                        blob_id, end_epoch, size, owner, source,
                        mime_type, extension, kind, actual_size,
                        is_active, fetch_ok, status, indexed, is_nsfw, nsfw_score,
                        first_seen_at, last_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(blob_id) DO UPDATE SET
                        end_epoch=COALESCE(excluded.end_epoch, blobs.end_epoch),
                        size=COALESCE(excluded.size, blobs.size),
                        owner=COALESCE(excluded.owner, blobs.owner),
                        mime_type=COALESCE(excluded.mime_type, blobs.mime_type),
                        extension=COALESCE(excluded.extension, blobs.extension),
                        kind=COALESCE(excluded.kind, blobs.kind),
                        actual_size=COALESCE(excluded.actual_size, blobs.actual_size),
                        fetch_ok=excluded.fetch_ok,
                        status=excluded.status,
                        indexed=excluded.indexed,
                        is_nsfw=excluded.is_nsfw,
                        nsfw_score=COALESCE(excluded.nsfw_score, blobs.nsfw_score),
                        last_updated_at=excluded.last_updated_at
                    """,
                    (
                        blob_id,
                        metadata_dict.get("expiresAt"),
                        size,
                        metadata_dict.get("owner"),
                        "vector_store",
                        mime_type,
                        metadata_dict.get("extension"),
                        metadata_dict.get("kind"),
                        size,
                        1,
                        1,
                        "indexed",
                        1,
                        1 if metadata_dict.get("is_nsfw") else 0,
                        metadata_dict.get("nsfw_score"),
                        now,
                        now,
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"[VectorStore] Warning: SQLite metadata sync failed ({blob_id}): {e}")

    def _hydrate_metadata_from_sqlite(self) -> None:
        """Load metadata from blob_catalog.sqlite for loaded embeddings when needed."""
        if not self.position_to_blob_id:
            return
        wanted = set(self.position_to_blob_id)
        if not wanted:
            return
        try:
            with sqlite3.connect(str(self.catalog_db_path), timeout=30) as conn:
                rows = conn.execute(
                    """
                    SELECT blob_id, size, owner, end_epoch, is_nsfw, mime_type, extension, kind
                    FROM blobs
                    WHERE indexed=1
                    """
                ).fetchall()
            loaded = 0
            for row in rows:
                blob_id = row[0]
                if blob_id not in wanted:
                    continue
                meta = {
                    "size": row[1],
                    "owner": row[2],
                    "expiresAt": row[3],
                    "is_nsfw": bool(row[4]),
                    "mime_type": row[5],
                    "extension": row[6],
                    "kind": row[7],
                    "is_quilt": False,
                    "parent_quilt_id": None,
                    "quilt_identifier": None,
                    "quilt_tags_json": None,
                }
                self.metadata[blob_id] = json.dumps(meta)
                loaded += 1
            if loaded:
                print(f"[VectorStore] Hydrated {loaded} metadata entries from SQLite")
        except Exception as e:
            print(f"[VectorStore] Warning: Could not hydrate metadata from SQLite: {e}")

    def _sync_all_metadata_to_sqlite(self) -> None:
        """Mirror in-memory metadata to SQLite to keep durable metadata always available."""
        if not self.metadata:
            return
        now = datetime.utcnow().isoformat()
        rows = []
        for blob_id, raw_meta in self.metadata.items():
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
            except Exception:
                continue
            size = int(meta.get("size") or 0)
            rows.append(
                (
                    blob_id,
                    meta.get("expiresAt"),
                    size,
                    meta.get("owner"),
                    "vector_store",
                    meta.get("mime_type"),
                    meta.get("extension"),
                    meta.get("kind"),
                    size,
                    1,
                    1,
                    "indexed",
                    1,
                    1 if meta.get("is_nsfw") else 0,
                    now,
                    now,
                )
            )
        if not rows:
            return
        try:
            with sqlite3.connect(str(self.catalog_db_path), timeout=30) as conn:
                conn.executemany(
                    """
                    INSERT INTO blobs (
                        blob_id, end_epoch, size, owner, source,
                        mime_type, extension, kind, actual_size,
                        is_active, fetch_ok, status, indexed, is_nsfw,
                        first_seen_at, last_updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(blob_id) DO UPDATE SET
                        end_epoch=COALESCE(excluded.end_epoch, blobs.end_epoch),
                        size=COALESCE(excluded.size, blobs.size),
                        owner=COALESCE(excluded.owner, blobs.owner),
                        mime_type=COALESCE(excluded.mime_type, blobs.mime_type),
                        extension=COALESCE(excluded.extension, blobs.extension),
                        kind=COALESCE(excluded.kind, blobs.kind),
                        actual_size=COALESCE(excluded.actual_size, blobs.actual_size),
                        fetch_ok=excluded.fetch_ok,
                        status=excluded.status,
                        indexed=excluded.indexed,
                        is_nsfw=excluded.is_nsfw,
                        last_updated_at=excluded.last_updated_at
                    """,
                    rows,
                )
                conn.commit()
        except Exception as e:
            print(f"[VectorStore] Warning: SQLite full metadata sync failed: {e}")
        
        try:
            meta = json.loads(self.metadata[blob_id])
            meta["blob_id"] = blob_id
            return meta
        except (json.JSONDecodeError, TypeError):
            if isinstance(self.metadata[blob_id], dict):
                meta = self.metadata[blob_id].copy()
                meta["blob_id"] = blob_id
                return meta
            return None

    def save(self, create_backup: bool = True) -> None:
        """Save the index, embeddings, metadata, and cursor to disk."""
        if not self._save_lock.acquire(blocking=False):
            print("[VectorStore] Save already in progress, skipping duplicate save")
            return

        try:
            # Apply persisted sweep blacklist before every save so the indexer
            # can never restore blobs that were removed by a sweep running in
            # another worker process.
            blacklist_path = self.index_path.parent / "deleted_blobs.json"
            try:
                if blacklist_path.exists():
                    with open(blacklist_path) as _blf:
                        blacklisted = set(json.load(_blf))
                    to_drop = [b for b in blacklisted if b in self.metadata or b in self.embeddings_dict]
                    if to_drop:
                        print(f"[VectorStore] Blacklist: removing {len(to_drop)} swept blobs before save")
                        self.remove_blob_ids(to_drop)
            except Exception as _bl_err:
                print(f"[VectorStore] Warning: could not apply sweep blacklist: {_bl_err}")

            if create_backup:
                try:
                    self._create_backup()
                except Exception as e:
                    print(f"[VectorStore] Warning: Backup creation failed: {e}")

            # Save Index
            if self._is_built and self.index is not None:
                try:
                    if self.backend == "faiss":
                        faiss.write_index(self.index, str(self.index_path))
                        print(f"[VectorStore] Saved FAISS index to {self.index_path}")
                    elif self.backend == "cuvs":
                        cagra.save(str(self.index_path), self.index)
                        print(f"[VectorStore] Saved cuVS index to {self.index_path}")
                except Exception as e:
                    print(f"[VectorStore] Warning: Failed to save index: {e}")
            
            # Save embeddings
            if self.embeddings_list:
                try:
                    embeddings_path = self.index_path.parent / "embeddings.npy"
                    embeddings_array = np.array(self.embeddings_list, dtype=np.float32)
                    np.save(str(embeddings_path), embeddings_array)
                    
                    mapping_path = self.index_path.parent / "blob_id_mapping.npy"
                    np.save(str(mapping_path), np.array(self.position_to_blob_id, dtype=object))
                    print(f"[VectorStore] Saved {len(self.embeddings_list)} embeddings")
                except Exception as e:
                    print(f"[VectorStore] Error saving embeddings: {e}")
            
            # Save metadata
            if self.metadata:
                try:
                    with open(self.metadata_path, 'w') as f:
                        json.dump(self.metadata, f, indent=2)
                    self._sync_all_metadata_to_sqlite()
                except Exception as e:
                    print(f"[VectorStore] Error saving metadata: {e}")
            
            # Save cursor
            try:
                self.cursor["last_updated"] = datetime.utcnow().isoformat()
                with open(self.cursor_path, 'w') as f:
                    json.dump(self.cursor, f, indent=2)
            except Exception as e:
                print(f"[VectorStore] Error saving cursor: {e}")
                
        finally:
            self._save_lock.release()

    def load(self) -> None:
        """Load the index, embeddings, metadata, and cursor from disk."""
        embeddings_path = self.index_path.parent / "embeddings.npy"
        mapping_path = self.index_path.parent / "blob_id_mapping.npy"
        
        embeddings_loaded = False
        skip_index_due_to_dim = False

        if embeddings_path.exists():
            try:
                embeddings_array = np.load(str(embeddings_path))
                if embeddings_array.size > 0 and embeddings_array.shape[-1] != self.embedding_dim:
                    print(
                        f"[VectorStore] Skipping load: embeddings dim {embeddings_array.shape[-1]} != "
                        f"embedding_dim {self.embedding_dim}. Use OMURA_VECTOR_INDEX_VERSION=omni and re-index."
                    )
                    self.embeddings_list = []
                    self.embeddings_dict = {}
                    self.position_to_blob_id = []
                    skip_index_due_to_dim = True
                    # Metadata will be skipped below to keep store consistent (fresh start).
                else:
                    self.embeddings_list = [embeddings_array[i] for i in range(len(embeddings_array))]
                    if mapping_path.exists():
                        blob_ids_array = np.load(str(mapping_path), allow_pickle=True)
                        self.position_to_blob_id = blob_ids_array.tolist()
                        for i, blob_id in enumerate(self.position_to_blob_id):
                            if i < len(self.embeddings_list):
                                self.embeddings_dict[blob_id] = self.embeddings_list[i]
                    print(f"[VectorStore] Loaded {len(self.embeddings_list)} embeddings")
                    embeddings_loaded = True
            except Exception as e:
                print(f"[VectorStore] Warning: Could not load embeddings: {e}")
                self.embeddings_list = []
                self.embeddings_dict = {}
                self.position_to_blob_id = []

        if self.metadata_path.exists() and not skip_index_due_to_dim:
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata_raw = json.load(f)
                self.metadata = {}
                for blob_id, meta_value in metadata_raw.items():
                    if isinstance(meta_value, str):
                        self.metadata[blob_id] = meta_value
                    else:
                        self.metadata[blob_id] = json.dumps(meta_value)
                print(f"[VectorStore] Loaded {len(self.metadata)} metadata entries")
            except Exception as e:
                print(f"[VectorStore] Warning: Could not load metadata: {e}")
                self.metadata = {}

        if self.cursor_path.exists() and not skip_index_due_to_dim:
            try:
                with open(self.cursor_path, 'r') as f:
                    self.cursor = json.load(f)
            except Exception as e:
                print(f"[VectorStore] Warning: Could not load cursor: {e}")

        if not self.metadata:
            self._hydrate_metadata_from_sqlite()

        # Try load index (skip if embeddings were skipped due to dimension mismatch)
        if self.index_path.exists() and not skip_index_due_to_dim:
            try:
                if self.backend == "faiss":
                    self.index = faiss.read_index(str(self.index_path))
                    print(f"[VectorStore] Loaded FAISS index from {self.index_path}")
                    self._is_built = True
                elif self.backend == "cuvs":
                    self.index = cagra.load(str(self.index_path))
                    print(f"[VectorStore] Loaded cuVS index from {self.index_path}")
                    self._is_built = True
            except Exception as e:
                print(f"[VectorStore] Warning: Could not load index: {e}")
                self.index = None
                self._is_built = False
                if embeddings_loaded and len(self.embeddings_list) > 0:
                    print("[VectorStore] Index will be built automatically on first search")
                    if self.backend == "faiss":
                        self.build_index()

        # Check consistency
        index_size = 0
        if self.index is not None:
            if self.backend == "faiss":
                index_size = self.index.ntotal
            # cagra index doesn't easily expose size property in python wrapper in all versions
        
        if self.backend == "faiss" and index_size != len(self.embeddings_list):
             print(f"[VectorStore] Warning: Index size ({index_size}) != embeddings ({len(self.embeddings_list)}). Rebuilding...")
             self.build_index()

    def _create_backup(self) -> None:
        """Create a backup of the current state."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / timestamp
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        files_to_backup = [
            (self.index_path.name, self.index_path),
            ("embeddings.npy", self.index_path.parent / "embeddings.npy"),
            ("blob_id_mapping.npy", self.index_path.parent / "blob_id_mapping.npy"),
            ("metadata.json", self.metadata_path),
            ("cursor.json", self.cursor_path),
        ]
        
        backed_up = 0
        for backup_name, source_path in files_to_backup:
            if source_path.exists():
                shutil.copy2(source_path, backup_subdir / backup_name)
                backed_up += 1
        
        if backed_up > 0:
            print(f"[VectorStore] Created backup at {backup_subdir}")

    def update_cursor(self, page: Optional[int] = None, last_processed_blob_id: Optional[str] = None) -> None:
        """Update the indexing cursor."""
        if page is not None:
            self.cursor["last_page"] = page
        if last_processed_blob_id is not None:
            self.cursor["last_processed_blob_id"] = last_processed_blob_id
        self.cursor["last_updated"] = datetime.utcnow().isoformat()

    def get_cursor(self) -> Dict:
        """Get the current cursor state."""
        return self.cursor.copy()

    def size(self) -> int:
        """Return the number of embeddings in the store."""
        return len(self.metadata)

    def counts_by_kind(self) -> Dict[str, int]:
        """Return counts of indexed items per modality (image, video, audio, doc, other)."""
        counts: Dict[str, int] = {
            "image": 0,
            "video": 0,
            "audio": 0,
            "doc": 0,
            "quilt": 0,
            "other": 0,
        }
        for blob_id, raw in self.metadata.items():
            try:
                meta = json.loads(raw) if isinstance(raw, str) else raw
                k = (meta.get("kind") or "other").lower()
            except (json.JSONDecodeError, TypeError, AttributeError):
                k = "other"
            if k == "image":
                counts["image"] += 1
            elif k == "video":
                counts["video"] += 1
            elif k == "audio":
                counts["audio"] += 1
            elif k == "quilt":
                counts["quilt"] += 1
            elif k in ("doc", "pdf", "text"):
                counts["doc"] += 1
            else:
                counts["other"] += 1
        return counts

    def get_blob_metadata(self, blob_id: str) -> Optional[Dict]:
        """Get metadata for a specific blob."""
        return self._get_metadata_by_blob_id(blob_id)

    def get_embedding(self, blob_id: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a specific blob.
        
        Args:
            blob_id: The blob ID to lookup
            
        Returns:
            Numpy array of the embedding, or None if not found
        """
        return self.embeddings_dict.get(blob_id)

    def prune_expired(self, current_epoch: int, dry_run: bool = False) -> Dict[str, int]:
        """Prune expired blobs from in-memory state and index.

        A blob is expired when metadata has `expiresAt` and `expiresAt <= current_epoch`.
        """
        total_before = len(self.metadata)
        expired_blob_ids: List[str] = []

        for blob_id, raw_meta in self.metadata.items():
            try:
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else raw_meta
            except (TypeError, json.JSONDecodeError):
                meta = {}

            expires_at = meta.get("expiresAt") if isinstance(meta, dict) else None
            if expires_at is None:
                continue

            try:
                if int(expires_at) <= current_epoch:
                    expired_blob_ids.append(blob_id)
            except (TypeError, ValueError):
                continue

        expired_count = len(expired_blob_ids)
        kept_count = total_before - expired_count
        summary = {
            "total_before": total_before,
            "expired_count": expired_count,
            "kept_count": kept_count,
        }

        if dry_run or expired_count == 0:
            return summary

        self.remove_blob_ids(expired_blob_ids)
        return summary

    def remove_blob_ids(self, blob_ids: List[str]) -> int:
        """Remove a set of blob_ids from metadata, embeddings, and index. Returns count removed.

        Caller is responsible for calling ``save()`` afterwards.
        """
        targets = set(b for b in blob_ids if b in self.embeddings_dict or b in self.metadata)
        if not targets:
            return 0

        for blob_id in targets:
            self.metadata.pop(blob_id, None)
            self.embeddings_dict.pop(blob_id, None)

        # Preserve stable ordering where possible.
        new_position_to_blob_id: List[str] = []
        new_embeddings_list: List[np.ndarray] = []

        for blob_id in self.position_to_blob_id:
            if blob_id in targets:
                continue
            embedding = self.embeddings_dict.get(blob_id)
            if embedding is None:
                continue
            new_position_to_blob_id.append(blob_id)
            new_embeddings_list.append(embedding)

        # Include any stragglers not present in position list to keep structures consistent.
        if len(new_position_to_blob_id) != len(self.embeddings_dict):
            for blob_id, embedding in self.embeddings_dict.items():
                if blob_id in new_position_to_blob_id:
                    continue
                new_position_to_blob_id.append(blob_id)
                new_embeddings_list.append(embedding)

        self.position_to_blob_id = new_position_to_blob_id
        self.embeddings_list = new_embeddings_list

        if self.backend == "cuvs":
            self.index = None
            self.gpu_embeddings = None

        if self.embeddings_list:
            self._is_built = False
            self.build_index()
        else:
            if self.backend == "faiss":
                self._init_faiss_index()
                self._is_built = True
            else:
                self.index = None
                self._is_built = False

        return len(targets)

    def clear(self) -> None:
        """Clear all embeddings and metadata."""
        self.metadata = {}
        self.embeddings_dict = {}
        self.embeddings_list = []
        self.position_to_blob_id = []
        
        if self.backend == "faiss":
            self._init_faiss_index()
        else:
            self.index = None
            
        self._is_built = False if self.backend == "cuvs" else True
        self.cursor = {
            "last_page": 0,
            "last_processed_blob_id": None,
            "last_updated": None,
        }

    def __del__(self) -> None:
        pass


__all__ = ["VectorStore", "EMBEDDING_DIM", "FAISS_AVAILABLE", "CUVS_AVAILABLE"]
