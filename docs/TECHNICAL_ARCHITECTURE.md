# Omura: Technical Architecture & Design

## Executive Summary

**Omura** is a multimodal search engine for the Walrus protocol that provides GPU-accelerated vector similarity search across blob content stored on the Sui blockchain. The system continuously discovers, indexes, and enables natural language search across images, text, audio, and video content using state-of-the-art embedding models and efficient vector search infrastructure.

### Walrus Protocol Overview

**Walrus** is a decentralized storage protocol built on the Sui blockchain that enables permanent, on-chain blob storage. Key characteristics:

- **Blob Storage**: Stores arbitrary binary data (images, videos, audio, documents, etc.) on-chain
- **Epoch-Based Lifecycle**: Blobs have a start epoch and end epoch defining their active period
- **Expiry System**: Blobs automatically expire when the current Sui epoch exceeds their `end_epoch`
- **Base64 Encoding**: Blob IDs are base64-encoded identifiers used for retrieval
- **Aggregator Network**: Distributed aggregator nodes provide HTTP access to blob content
- **Epoch Tracking**: Sui epochs increment periodically, enabling time-based blob expiration

**Walrus Blob Lifecycle:**
1. **Registration**: Blob uploaded to Walrus with `start_epoch` and `end_epoch`
2. **Active Period**: Blob is accessible while `current_epoch < end_epoch`
3. **Expiration**: Blob becomes inaccessible when `current_epoch >= end_epoch`
4. **Deletion**: Expired blobs can be deleted from storage (reclaiming space)

**Omura's Role**: Indexes active Walrus blobs for searchable discovery across the entire protocol, enabling users to find content without knowing blob IDs.

---

## System Architecture Overview

```mermaid
graph TB
    subgraph "External Services"
        BB[Blockberry API<br/>Walrus Blob Discovery]
        WA[Walrus Aggregator<br/>Blob Content Storage]
        SUI[Sui Blockchain<br/>Epoch Tracking]
    end
    
    subgraph "Omura Core System"
        subgraph "API Layer"
            API[FastAPI Server<br/>Port 19353]
            SR[Search Routes<br/>/search/]
            BR[Blob Routes<br/>/blob/]
        end
        
        subgraph "Indexing Layer"
            INDEXER[Multimodal Indexer<br/>Background Thread]
            BB_CLIENT[Blockberry Client<br/>Streaming Iterator]
            CURSOR[Cursor Manager<br/>Pagination State]
            DIST_INDEXER[Distributed Indexer<br/>Multi-node Support]
            EVENT_HANDLER[Event Handler<br/>Real-time Updates]
        end
        
        subgraph "Processing Layer"
            EMB[Embedding Generator<br/>Omura Embed / BGE / CLAP]
            FD[File Detector<br/>Magic Bytes]
            NSFW[NSFW Classifier<br/>Zero-shot]
            EPOCH[Epoch Tracker<br/>Blob Expiry Filter]
        end
        
        subgraph "Storage Layer"
            VS[Vector Store<br/>FAISS/cuVS]
            DISK[(Persistent Index<br/>data/vector_index/)]
        end
    end
    
    SUI -->|Epoch Updates| EPOCH
    BB -->|Discover Blobs| BB_CLIENT
    SUI -->|Epoch Events| EVENT_HANDLER
    EVENT_HANDLER -->|Real-time Blob Events| INDEXER
    BB_CLIENT -->|Stream Blob IDs<br/>with Metadata| EPOCH
    EPOCH -->|Filter Active Blobs<br/>end_epoch > current_epoch| INDEXER
    WA -->|Fetch Content| INDEXER
    DIST_INDEXER -->|Shared State| INDEXER
    INDEXER -->|Detect Type| FD
    INDEXER -->|Generate Embeddings| EMB
    INDEXER -->|Video-Audio Fusion| VIDEO_FUSION
    INDEXER -->|Check Safety| NSFW
    INDEXER -->|Store Embeddings| VS
    VS <-->|Load/Save| DISK
    CURSOR <-->|Track Progress| INDEXER
    
    API -->|Read| VS
    API -->|Proxy| WA
    SR -->|Search| VS
    BR -->|Fetch & Detect| WA
    
    INDEXER -.->|Shared Store| VS
    SR -.->|Shared Store| VS
    
    style EPOCH fill:#ffe1e1
    style CURSOR fill:#e1ffe1
    style FD fill:#e1f5ff
```

---

## Indexing Pipeline Architecture

### Discovery & Filtering Flow

```mermaid
sequenceDiagram
    participant BB as Blockberry API
    participant Epoch as Epoch Tracker
    participant Indexer as Multimodal Indexer
    participant Detector as File Detector
    participant Embedding as Embedding Generator
    participant VectorStore as Vector Store
    
    loop Continuous Indexing + Real-time Updates
        BB->>Epoch: Query current epoch<br/>(infer from recent blobs)
        Epoch->>Epoch: Determine current_epoch<br/>(or use MANUAL_EPOCH=21)
        
        par Batch Indexing
            BB->>Indexer: Stream blob IDs (paginated)<br/>with metadata (end_epoch, size, owner)
        and Real-time Events
            EVENT_HANDLER->>Indexer: New blob events (WebSocket/subscription)<br/>incremental updates
        end
        
        loop For each blob
            Indexer->>Epoch: Check: end_epoch > current_epoch?
            alt Blob Active
                Epoch->>Indexer: Blob is active → Process
                Indexer->>Indexer: Fetch blob content from Aggregator
                Indexer->>Detector: Detect file type (magic bytes)
                
                alt Image Detected
                    Detector-->>Indexer: (mime_type, extension, kind="image")
                    Indexer->>Embedding: Generate image embedding (Omura Embed, 768-dim)
                    Embedding-->>Indexer: Normalized embedding vector
                else Text Detected
                    Detector-->>Indexer: (mime_type, extension, kind="text")
                    Indexer->>Embedding: Generate text embedding (BGE, 2048-dim)
                    Embedding-->>Indexer: Normalized embedding vector
                else Audio Detected
                    Detector-->>Indexer: (mime_type, extension, kind="audio")
                    Indexer->>Embedding: Generate audio embedding (CLAP, 512-dim)
                    Embedding-->>Indexer: Normalized embedding vector
                else Video Detected
                    Detector-->>Indexer: (mime_type, extension, kind="video")
                    Indexer->>Embedding: Extract video frames + audio track
                    Indexer->>Embedding: Generate video embedding (EgoVLPv2, 768-dim)
                    Indexer->>Embedding: Generate audio embedding (CLAP, 512-dim)
                    Indexer->>Embedding: Concatenate [video_emb, audio_emb] → 1280-dim
                    Indexer->>Embedding: Apply Video Fusion MLP → 2048-dim
                    Embedding-->>Indexer: Normalized hybrid embedding vector
                end
                
                Indexer->>NSFW: Check content safety (zero-shot classification)
                NSFW-->>Indexer: is_nsfw flag (similarity >= 0.55)
                Indexer->>Indexer: Extract additional metadata<br/>(EXIF, ID3, video metadata)
                Indexer->>Indexer: Quality scoring (content quality metrics)
                
                Indexer->>VectorStore: Add embedding + enriched metadata<br/>(blob_id, size, end_epoch, is_nsfw, quality_score, extracted_metadata)
                Indexer->>DuckDB: Store metadata in blobs table<br/>(with enriched fields)
                Indexer->>Indexer: Update cursor (page, last_blob_id)
            else Blob Expired
                Epoch->>Indexer: Skip blob (end_epoch <= current_epoch)
                Note over Indexer: Blob excluded from indexing
            end
        end
        
        Note over Indexer,VectorStore: Every 2 batches: Rebuild index<br/>Every 5 batches: Save to disk<br/>Every batch: Update cursor
    end
```

---

## File Type Recognition System

### Magic Bytes Detection Pipeline

```mermaid
graph LR
    subgraph "File Type Detection"
        INPUT[Blob Content<br/>Raw Bytes]
        MB[Magic Bytes<br/>First 2048 bytes]
        LIB[python-magic<br/>libmagic Library]
        
        subgraph "Detection Methods"
            MIME[MIME Type Detection]
            EXT[Extension Mapping]
            KIND[Content Kind<br/>image/text/audio/video/pdf]
        end
        
        subgraph "Fallback Heuristics"
            PDF[PDF Header<br/>%PDF-]
            PNG[PNG Header<br/>89 50 4E 47]
            JPEG[JPEG Header<br/>FF D8 FF]
            GIF[GIF Header<br/>GIF87a/GIF89a]
            WEBP[WebP Header<br/>RIFF...WEBP]
            ZIP[ZIP Header<br/>PK 03 04]
        end
    end
    
    INPUT --> MB
    MB --> LIB
    LIB --> MIME
    MIME --> EXT
    EXT --> KIND
    
    LIB -.->|If magic fails| PDF
    PDF -.->|Fallback| MIME
    PNG -.->|Fallback| MIME
    JPEG -.->|Fallback| MIME
    GIF -.->|Fallback| MIME
    WEBP -.->|Fallback| MIME
    ZIP -.->|Fallback| MIME
    
    style LIB fill:#e1f5ff
    style MIME fill:#e1ffe1
    style KIND fill:#ffe1e1
```

### File Type Detection Algorithm

```python
# Simplified detection flow
def detect_file_type(data: bytes) -> Tuple[str, str, str]:
    """
    Returns: (mime_type, extension, kind)
    
    Process:
    1. Use python-magic (libmagic) for comprehensive detection (first 2048 bytes)
    2. Map MIME type to extension via lookup table
    3. Determine content kind category (image/text/audio/video/pdf/archive/binary)
    4. Fallback to heuristics if magic fails
    """
```

**Supported File Types:**
- **Images**: PNG, JPEG, GIF, WebP, BMP, TIFF, SVG
- **Text**: Plain text, HTML, CSS, JavaScript, JSON, XML, Python
- **Audio**: MP3, WAV, OGG, FLAC
- **Video**: MP4, WebM, MOV, AVI, MKV
- **Documents**: PDF, DOC, DOCX, XLS, PPT
- **Archives**: ZIP, TAR, GZ, BZ2, 7Z, RAR
- **Binary**: Fallback to `application/octet-stream`

**Advanced File Parsing:**
- **PDF Text Extraction**: Deep content extraction using PyPDF (text content from PDFs)
- **Video Frame Sampling**: Intelligent keyframe extraction for video processing
- **Audio Transcription**: Speech-to-text capabilities for audio content (future: Whisper integration)
- **Metadata Extraction**: EXIF data from images, ID3 tags from audio, container metadata from videos
- **Content Chunking**: Split large documents into chunks for better embedding quality

---

## Epoch Tracking & Blob Expiry System

### Epoch-Based Expiry Architecture

```mermaid
stateDiagram-v2
    [*] --> DiscoverBlobs: Start Indexing
    
    DiscoverBlobs --> GetCurrentEpoch: Query Blockberry API
    
    GetCurrentEpoch --> InferEpoch: Analyze recent blobs
    InferEpoch --> UseManualEpoch: Fallback if inference fails
    GetCurrentEpoch --> UseManualEpoch: Use MANUAL_EPOCH=21
    
    UseManualEpoch --> StreamBlobs: current_epoch set
    
    StreamBlobs --> CheckExpiry: For each blob<br/>(page, blob_id, metadata)
    
    CheckExpiry --> Active: end_epoch > current_epoch
    CheckExpiry --> Expired: end_epoch <= current_epoch
    
    Active --> IndexBlob: Process blob
    Expired --> SkipBlob: Exclude from index
    
    IndexBlob --> UpdateCursor: Store embedding
    SkipBlob --> StreamBlobs: Continue
    
    UpdateCursor --> StreamBlobs: Track progress
    
    StreamBlobs --> EpochUpdate: Periodically check<br/>for epoch changes
    EpochUpdate --> GetCurrentEpoch: Re-evaluate current epoch
    
    StreamBlobs --> [*]: End of stream<br/>(wait and retry)
    
    note right of Expired
        Blobs with end_epoch <= current_epoch
        are never indexed. They are filtered
        during discovery phase.
    end note
    
    note right of Active
        Only active blobs (end_epoch > current_epoch)
        are processed and stored in vector index.
    end note
```

### Epoch Detection & Filtering

**Implementation:**
- **Manual Epoch Override**: `MANUAL_EPOCH = 21` (configurable fallback)
- **Automatic Inference**: Queries recent blobs from Blockberry API to infer current epoch
- **Real-time Epoch Tracking**: Monitor Sui blockchain for epoch changes via WebSocket subscriptions
- **Instant Updates**: Automatic re-evaluation of blob expiry on epoch changes
- **Filtering Logic**: `is_active = end_epoch > current_epoch`

**Epoch Metadata:**
- `startEpoch` / `start_epoch`: Epoch when blob was registered (immutable)
- `endEpoch` / `end_epoch` / `expiresAtEpoch`: Epoch when blob expires (defines active period)
- `current_epoch`: Inferred from recent blob metadata (most common start_epoch) or updated via WebSocket

**Real-time Epoch Tracking:**
```python
# WebSocket subscription to Sui blockchain
subscribe_to_epoch_changes():
    on_epoch_change(new_epoch):
        current_epoch = new_epoch
        # Trigger re-evaluation of blob expiry status
        re_evaluate_expired_blobs()
        # Update cursor with new epoch in DuckDB
        update_cursor(current_epoch=new_epoch)
```

**Expiry Handling:**
```python
# During blob discovery
for blob in stream_blobs():
    end_epoch = blob.get("endEpoch") or blob.get("end_epoch")
    is_active = int(end_epoch) > current_epoch
    
    if is_active:
        # Process and index blob
        index_blob(blob)
    else:
        # Skip expired blob
        continue
```

---

## Data Updates & Cursor Management

### Incremental Indexing with Cursor Tracking

```mermaid
graph TB
    subgraph "Cursor-Based Indexing"
        START[Start/Resume Indexing]
        LOAD[Load Cursor from DuckDB<br/>cursor table]
        CHECK{Cursor Exists?}
        
        NEW[Start from page 0]
        RESUME[Resume from<br/>last_page, last_blob_id]
        
        STREAM[Stream Blobs<br/>from last_page]
        BATCH[Batch Processing<br/>default: 100 blobs]
        PROCESS[Process Blob]
        
        UPDATE_CURSOR[Update Cursor<br/>page, last_blob_id]
        SAVE_CURSOR[Save Cursor<br/>to DuckDB cursor table]
        
        REBUILD[Rebuild Index<br/>Every 2 batches]
        PERSIST[Save to Disk<br/>Every 5 batches]
    end
    
    START --> LOAD
    LOAD --> CHECK
    CHECK -->|No cursor| NEW
    CHECK -->|Cursor exists| RESUME
    
    NEW --> STREAM
    RESUME --> STREAM
    
    STREAM --> BATCH
    BATCH --> PROCESS
    PROCESS --> UPDATE_CURSOR
    UPDATE_CURSOR --> SAVE_CURSOR
    
    SAVE_CURSOR --> REBUILD
    REBUILD --> PERSIST
    PERSIST --> STREAM
    
    style LOAD fill:#e1ffe1
    style RESUME fill:#ffe1e1
    style SAVE_CURSOR fill:#e1ffe1
```

### Cursor Structure

**Stored in DuckDB `cursor` table:**
```sql
-- Singleton table (only one row)
SELECT last_page, last_processed_blob_id, current_epoch, last_updated
FROM cursor
WHERE id = 1;
```

**Structure:**
- `id`: Always 1 (singleton constraint)
- `last_page`: Last processed page number (for paginated indexing)
- `last_processed_blob_id`: Last successfully indexed blob ID (for resumption)
- `current_epoch`: Current Walrus epoch (cached, updated via WebSocket)
- `last_updated`: Timestamp of last update

### Distributed Indexing Support

**Multi-Node Architecture:**
- **Shared DuckDB State**: All indexer nodes share the same DuckDB database for cursor state
- **Coordinated Indexing**: File locking ensures only one node processes each page
- **Distributed Vector Stores**: Each node maintains its own FAISS/cuVS index (can be merged)
- **Load Balancing**: Multiple indexer nodes can run in parallel with shared cursor tracking

**Incremental Updates:**
- **Event-Driven Indexing**: Real-time indexing of new blobs as they appear via WebSocket subscriptions
- **Batch + Real-time**: Combines batch pagination with real-time event processing
- **Incremental Sync**: Nodes sync cursor state and processed blob IDs periodically

**Cursor Functions:**
- **Resume Capability**: Indexer can resume from exact position after crash/restart
- **Skip Tracking**: When resuming, skips blobs until `last_processed_blob_id` is found
- **Progress Tracking**: Enables monitoring of indexing progress
- **Atomic Updates**: Cursor updated after each batch completion

### Update Strategy

```mermaid
sequenceDiagram
    participant Indexer
    participant Cursor
    participant VectorStore
    participant Disk
    
    Note over Indexer: Indexing Loop
    
    Indexer->>DuckDB: Query cursor table<br/>SELECT * FROM cursor WHERE id=1
    DuckDB-->>Indexer: last_page=42, last_blob_id="ABC123"
    
    Indexer->>Indexer: Stream from page 42<br/>Skip until blob_id="ABC123"
    
    loop Process batches
        Indexer->>VectorStore: Add embeddings (batch of 100)
        Indexer->>DuckDB: Update cursor table<br/>UPDATE cursor SET last_page=45,<br/>last_blob_id="XYZ789" WHERE id=1
        Indexer->>DuckDB: Commit transaction
        
        alt Every 2 batches
            Indexer->>VectorStore: Rebuild index
        end
        
        alt Every 5 batches
            Indexer->>VectorStore: Save index + embeddings<br/>Create backup
            VectorStore->>Disk: Write files
        end
    end
    
    Note over Indexer: End of stream reached
    Indexer->>Indexer: Wait 30s, retry from last_page
```

**Update Frequency:**
- **Cursor**: Updated after every batch (real-time progress tracking)
- **Index Rebuild**: Every 2 batches (~200 blobs)
- **Persistent Save**: Every 5 batches (~500 blobs)
- **Backups**: Created on every save (timestamped directories)

---

## Blob Expiry & Cleanup Strategy

### Current Expiry Handling

```mermaid
graph LR
    subgraph "Expiry Prevention (Current)"
        DISCOVER[Blob Discovery]
        FILTER[Filter: end_epoch > current_epoch]
        INDEX[Index Active Blobs Only]
    end
    
    subgraph "Future: Cleanup Strategy"
        PERIODIC[Periodic Cleanup Task]
        CHECK[Check Stored Blobs<br/>end_epoch <= current_epoch]
        REMOVE[Remove from Vector Store]
        REBUILD[Rebuild Index]
    end
    
    DISCOVER --> FILTER
    FILTER --> INDEX
    
    PERIODIC --> CHECK
    CHECK --> REMOVE
    REMOVE --> REBUILD
    
    style FILTER fill:#e1ffe1
    style REMOVE fill:#ffe1e1
```

### Expiry Metadata Storage

**Stored in DuckDB `blobs` table:**
```sql
SELECT blob_id, size, owner, expiresAt, is_nsfw, indexed_at
FROM blobs
WHERE expiresAt <= current_epoch;
```

**Expiry Handling Strategy:**
- ✅ **Prevention**: Expired blobs are filtered during discovery (never indexed)
- ✅ **Automatic Cleanup**: Cron-based cleanup job removes expired blobs from index
- ✅ **Periodic Updates**: Epoch updates trigger re-evaluation of blob expiry status
- ✅ **SQL Queries**: Efficient expiry filtering using DuckDB indexed queries

### Epoch Update Handling

```mermaid
sequenceDiagram
    participant EpochTracker
    participant Blockberry
    participant Indexer
    participant VectorStore
    
    Note over EpochTracker: On Startup / Periodically
    
    EpochTracker->>Blockberry: Fetch recent blobs<br/>(page 0, size 100, DESC)
    Blockberry-->>EpochTracker: Recent blob metadata
    
    EpochTracker->>EpochTracker: Extract startEpoch values
    EpochTracker->>EpochTracker: Calculate most_common_start_epoch
    EpochTracker->>EpochTracker: current_epoch = most_common_start
    
    alt current_epoch changed
        EpochTracker->>Indexer: Update current_epoch
        Note over Indexer: Future blobs filtered<br/>with new epoch value
        Indexer->>VectorStore: Continue indexing<br/>only active blobs
    else current_epoch unchanged
        Note over Indexer: Continue with<br/>existing epoch filter
    end
```

---

## Cron-Based Index Cleanup & Maintenance

### Cleanup Job Architecture

```mermaid
graph TB
    subgraph "Cron Scheduler"
        CRON[Cron Job<br/>Periodic Execution<br/>Default: Hourly]
    end
    
    subgraph "Cleanup Process"
        TRIGGER[Trigger Cleanup Job]
        GET_EPOCH[Get Current Epoch<br/>from Blockberry API]
        QUERY_DB[Query DuckDB<br/>SELECT blob_id, expiresAt<br/>FROM blobs WHERE expiresAt <= current_epoch]
        FETCH_EXPIRED[Get Expired Blob IDs]
        
        subgraph "Removal Process"
            REMOVE_FAISS[Remove from FAISS/cuVS<br/>Remove embedding from vector index]
            REMOVE_NUMPY[Remove from NumPy arrays<br/>Update embeddings.npy]
            REMOVE_DB[Delete from DuckDB<br/>DELETE FROM blobs, embeddings WHERE blob_id = ?]
            UPDATE_MAPPING[Update position_to_blob_id mapping<br/>Regenerate blob_id_mapping.npy]
        end
        
        REBUILD[Rebuild Vector Index]
        SAVE[Save Updated Index<br/>to Disk]
    end
    
    CRON --> TRIGGER
    TRIGGER --> GET_EPOCH
    GET_EPOCH --> QUERY_DB
    QUERY_DB --> FETCH_EXPIRED
    FETCH_EXPIRED --> REMOVE_FAISS
    REMOVE_FAISS --> REMOVE_NUMPY
    REMOVE_NUMPY --> REMOVE_DB
    REMOVE_DB --> UPDATE_MAPPING
    UPDATE_MAPPING --> REBUILD
    REBUILD --> SAVE
    
    style CRON fill:#ffe1e1
            style QUERY_DB fill:#e1f5ff
            style REMOVE_FAISS fill:#ffe1e1
            style REBUILD fill:#e1ffe1
```

### Cleanup Job Flow

```mermaid
sequenceDiagram
    participant Cron as Cron Scheduler
    participant Cleanup as Cleanup Job
    participant Blockberry as Blockberry API
    participant DuckDB as DuckDB Database<br/>(Metadata)
    participant FAISS as FAISS/cuVS<br/>(Vector DB)
    participant NumPy as NumPy Arrays<br/>(Embeddings)
    participant Disk as Disk Storage
    
    Note over Cron: Every hour (configurable)
    
    Cron->>Cleanup: Trigger cleanup job
    Cleanup->>Blockberry: Get current_epoch
    Blockberry-->>Cleanup: current_epoch = 22
    
    Cleanup->>DuckDB: Query expired blobs<br/>SELECT blob_id FROM blobs<br/>WHERE expiresAt <= 22
    DuckDB-->>Cleanup: List of expired blob_ids
    
    alt Expired blobs found
        loop For each expired blob
            Cleanup->>FAISS: Remove blob_id from index<br/>(remove embedding from FAISS/cuVS)
            Cleanup->>NumPy: Remove embedding from arrays<br/>(update embeddings.npy)
            Cleanup->>DuckDB: DELETE FROM blobs<br/>WHERE blob_id = ?
            Cleanup->>DuckDB: DELETE FROM embeddings<br/>WHERE blob_id = ?
            DuckDB-->>Cleanup: Confirmation
        end
        
        Cleanup->>FAISS: Rebuild vector index<br/>(rebuild FAISS/cuVS with remaining embeddings)
        Cleanup->>NumPy: Update position mappings<br/>(regenerate blob_id_mapping.npy)
        Cleanup->>Disk: Save updated FAISS index<br/>Save updated NumPy arrays
        Cleanup->>Disk: Save DuckDB database<br/>Create backup
        
        Cleanup->>Cleanup: Log cleanup stats<br/>blobs removed, timestamp
    else No expired blobs
        Note over Cleanup: Skip cleanup,<br/>index is up-to-date
    end
    
    Cleanup-->>Cron: Job completed
```

### Cleanup Job Configuration

**Schedule:**
- **Default**: Hourly (every 1 hour)
- **Configurable**: Via `OMURA_CLEANUP_INTERVAL_HOURS` environment variable
- **Alternative**: Cron expression for custom schedules

**Cleanup Process:**
1. **Epoch Detection**: Query Blockberry API for current epoch
2. **Database Query**: Query DuckDB for expired blobs (`expiresAt <= current_epoch`)
3. **Batch Removal**: Remove expired blobs from vector store (in-memory)
4. **Database Update**: Delete expired blob records from DuckDB
5. **Index Rebuild**: Rebuild FAISS/cuVS index with remaining embeddings
6. **Persistence**: Save updated index to disk with backup

**Performance:**
- **Batch Processing**: Process expired blobs in batches (configurable batch size)
- **Incremental Rebuild**: Only rebuild index after removals (optimize if no changes)
- **Locking**: Acquire lock during cleanup to prevent concurrent modifications

### Index Update Strategy

```mermaid
graph LR
    subgraph "Update Triggers"
        CRON_CLEANUP[Cron Cleanup Job<br/>Hourly]
        EPOCH_CHANGE[Epoch Change<br/>Detected]
        MANUAL[Manual Trigger<br/>API/CLI]
    end
    
    subgraph "Update Process"
        CHECK{Changes<br/>Detected?}
        REMOVE[Remove Expired]
        UPDATE[Update Metadata]
        REBUILD[Rebuild Index]
        SAVE[Persist Changes]
    end
    
    CRON_CLEANUP --> CHECK
    EPOCH_CHANGE --> CHECK
    MANUAL --> CHECK
    
    CHECK -->|Expired blobs| REMOVE
    CHECK -->|Metadata updates| UPDATE
    CHECK -->|No changes| SAVE
    
    REMOVE --> REBUILD
    UPDATE --> REBUILD
    REBUILD --> SAVE
    
    style CRON_CLEANUP fill:#ffe1e1
    style REBUILD fill:#e1ffe1
```

### Cron Job Implementation

**Job Types:**
1. **Cleanup Job**: Remove expired blobs (primary)
2. **Epoch Sync Job**: Update current epoch periodically (runs more frequently)
3. **Index Optimization Job**: Periodic index optimization/compaction (runs daily)

**Error Handling:**
- **Retry Logic**: Automatic retry on failures (exponential backoff)
- **Logging**: Comprehensive logging of cleanup operations
- **Metrics**: Track cleanup statistics (blobs removed, duration, errors)
- **Alerting**: Notify on cleanup failures or anomalies

---

## Storage Architecture: DuckDB Integration

### Database Schema Design

```mermaid
erDiagram
    BLOBS ||--o{ EMBEDDINGS : has
    BLOBS ||--o{ METADATA : has
    
    BLOBS {
        string blob_id PK
        int size
        string mime_type
        string extension
        string kind
        string owner
        int start_epoch
        int expiresAt
        bool is_nsfw
        datetime indexed_at
        datetime updated_at
    }
    
    EMBEDDINGS {
        string blob_id FK
        int embedding_dim
        array embeddings
        datetime created_at
    }
    
    METADATA {
        string blob_id FK
        json additional_metadata
        datetime updated_at
    }
```

### DuckDB-Based Storage Architecture

```mermaid
graph TB
    subgraph "Vector Store Structure"
        VS[VectorStore Instance<br/>In-Memory]
        
        subgraph "In-Memory Data"
            ED[embeddings_dict<br/>blob_id → vector]
            EL[embeddings_list<br/>Ordered vectors]
            PM[position_to_blob_id<br/>Index → blob_id]
        end
        
        subgraph "DuckDB Database"
            DB[(DuckDB Database<br/>omura.duckdb)]
            BLOBS_TABLE[blobs table<br/>blob_id, metadata]
            EMBEDDINGS_TABLE[embeddings table<br/>blob_id, embedding_ref]
            CURSOR_TABLE[cursor table<br/>last_page, last_blob_id]
        end
        
        subgraph "Disk Persistence"
            IDX[vector_index.faiss/cagra]
            EMB[embeddings.npy]
            MAP[blob_id_mapping.npy]
            DB_FILE[omura.duckdb]
        end
        
        subgraph "Backup System"
            BK[backups/YYYYMMDD_HHMMSS/]
            BK_IDX[vector_index.*]
            BK_EMB[embeddings.npy]
            BK_DB[omura.duckdb]
        end
    end
    
    VS --> ED
    VS --> EL
    VS --> PM
    
    VS <-->|Query/Update| DB
    DB --> BLOBS_TABLE
    DB --> EMBEDDINGS_TABLE
    DB --> CURSOR_TABLE
    
    VS <-->|Save/Load| IDX
    VS <-->|Save/Load| EMB
    VS <-->|Save/Load| MAP
    DB <-->|Save/Load| DB_FILE
    
    VS -->|Create Backup| BK
    BK --> BK_IDX
    BK --> BK_EMB
    BK --> BK_DB
    
    style VS fill:#e1f5ff
    style DB fill:#e1ffe1
    style BK fill:#ffe1e1
```

### DuckDB Schema

The DuckDB database schema defines three main tables for storing blob metadata, embedding references, and indexing state:

```sql
-- Main blobs table
-- Stores blob metadata including expiration, ownership, and content type
CREATE TABLE blobs (
    blob_id VARCHAR PRIMARY KEY,
    size INTEGER NOT NULL,
    mime_type VARCHAR,
    extension VARCHAR,
    kind VARCHAR,  -- image, text, audio, video, etc.
    owner VARCHAR,
    start_epoch INTEGER,
    expiresAt INTEGER,  -- end_epoch
    is_nsfw BOOLEAN DEFAULT FALSE,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Embeddings reference table
-- Maps blob_ids to embedding positions and dimensions
CREATE TABLE embeddings (
    blob_id VARCHAR PRIMARY KEY REFERENCES blobs(blob_id) ON DELETE CASCADE,
    embedding_dim INTEGER NOT NULL,  -- 768, 2048, etc.
    position INTEGER,  -- Position in embeddings_list
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cursor/Indexing state table
-- Singleton table storing pagination state for incremental indexing
CREATE TABLE cursor (
    id INTEGER PRIMARY KEY DEFAULT 1,  -- Singleton row (always 1)
    last_page INTEGER DEFAULT 0,
    last_processed_blob_id VARCHAR,
    current_epoch INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (id = 1)  -- Ensure only one row exists
);

-- Performance indexes
-- Optimize queries for expiry checks, modality filtering, and embedding lookups
CREATE INDEX idx_blobs_expiresAt ON blobs(expiresAt);
CREATE INDEX idx_blobs_kind ON blobs(kind);
CREATE INDEX idx_blobs_indexed_at ON blobs(indexed_at);
CREATE INDEX idx_embeddings_position ON embeddings(position);
CREATE INDEX idx_embeddings_blob_id ON embeddings(blob_id);
```

**Table Relationships:**
- `blobs` table: Primary metadata storage (one row per blob)
- `embeddings` table: Foreign key to `blobs.blob_id` (one-to-one relationship)
- `cursor` table: Singleton pattern (only one row with `id = 1`)
- Foreign key cascade: Deleting a blob automatically removes its embedding reference

### Storage Strategy

**Hybrid Storage Approach:**

The system uses a **hybrid architecture** where each component serves its optimal purpose:

- **FAISS/cuVS (Vector Database)**: 
  - Primary vector search engine for similarity search
  - Stores and indexes embedding vectors (768/2048-dim)
  - Handles fast k-NN search operations
  - Disk-persisted index files (`.faiss` or `.cagra`)
  - **Purpose**: Vector similarity search (images, text, audio, video)

- **DuckDB (Metadata & State Database)**:
  - Stores blob metadata (size, mime_type, owner, epochs, etc.)
  - Manages embedding references and position mappings
  - Tracks cursor state and indexing progress
  - Handles SQL queries for filtering, expiry checks, analytics
  - **Purpose**: Metadata storage, SQL queries, ACID transactions

- **NumPy Arrays (Embedding Storage)**:
  - Memory-mapped `.npy` files for embedding vectors
  - Used for loading embeddings into FAISS/cuVS
  - Efficient binary storage format
  - **Purpose**: Raw embedding vector storage

**Clear Separation of Concerns:**
- **Vector Search** → FAISS/cuVS (fast similarity search)
- **Metadata Queries** → DuckDB (SQL queries, filtering, analytics)
- **Raw Embeddings** → NumPy arrays (efficient storage)

**Benefits of Hybrid Approach:**
- ✅ **FAISS/cuVS**: Optimized for vector similarity search (specialized vector DB)
- ✅ **DuckDB**: Optimized for SQL queries and metadata operations (specialized SQL DB)
- ✅ **Best of Both**: Each database does what it's best at
- ✅ **No Conflicts**: FAISS handles vectors, DuckDB handles metadata

**Benefits of DuckDB for Metadata:**
- ✅ **SQL Queries**: Efficient querying of blob metadata, expiry filtering
- ✅ **ACID Transactions**: Reliable data consistency
- ✅ **Embedded Database**: No separate server required
- ✅ **High Performance**: Optimized columnar storage for analytics
- ✅ **Parquet Export**: Easy data export/backup
- ✅ **JSON Support**: Store complex metadata in JSON columns

**Data Flow:**
1. **Indexing**: 
   - Generate embeddings → Store in NumPy arrays
   - Build FAISS/cuVS index from NumPy arrays
   - Store blob metadata → DuckDB
   - Store embedding references → DuckDB

2. **Querying**: 
   - Search query → Generate query embedding
   - Vector search in FAISS/cuVS → Get top-k blob_ids with similarities
   - Query DuckDB with blob_ids → Get full metadata for results

3. **Cleanup**: 
   - Query DuckDB for expired blobs (SQL: `WHERE expiresAt <= current_epoch`)
   - Remove expired blob_ids from FAISS/cuVS index
   - Remove from NumPy arrays
   - Delete records from DuckDB
   - Rebuild FAISS/cuVS index

### Save/Load Cycle

**Save Triggers:**
- Every 5 batches during indexing
- Periodic save task (every 10 minutes)
- After cron cleanup job completion
- On graceful shutdown
- On error recovery (after max retries)

**Load Triggers:**
- On API server startup
- After indexer crash/restart
- After cron cleanup job (reload updated state)
- Manual reload via API/CLI

**Backup Strategy:**
- **Timestamped Directories**: `backups/20240115_103000/`
- **Complete State Snapshot**: DuckDB database + FAISS/cuVS index + NumPy embeddings
- **Automated Backup Rotation**: Automatic cleanup of old backups (configurable retention policy)
- **Backup Before Cleanup**: Create backup before cron cleanup jobs (safety net)
- **Incremental Backups**: Incremental backup support for large datasets
- **Backup Archival**: Automated archival strategies (local, cloud, long-term storage)
- **Backup Validation**: Automatic backup integrity checks

---

## Multimodal Embedding Architecture

### Complete Modality Support

Omura supports four primary modalities with specialized encoders and unified projection to a 2048-dim shared space:

#### 1. Image Modality

**Encoder**: Omura Embed ViT-B-16
- **Input**: Image bytes (PNG, JPEG, GIF, WebP)
- **Embedding Dimension**: 768-dim
- **Projection**: Image MLP (768 → 2048-dim)
- **Features**: Vision-language alignment, text-to-image search

#### 2. Text Modality

**Encoder**: BGE (BAAI General Embedding) Variant
- **Input**: Text strings (natural language queries, documents)
- **Embedding Dimension**: 2048-dim (native)
- **Projection**: Text MLP (2048 → 2048-dim, identity/normalization)
- **Features**: High-quality text embeddings, multilingual support

#### 3. Audio Modality

**Encoder**: CLAP (Contrastive Language-Audio Pre-training)
- **Input**: Audio bytes (MP3, WAV, OGG, FLAC)
- **Embedding Dimension**: 512-dim (native)
- **Projection**: Audio MLP (512 → 2048-dim, up-projection)
- **Features**: Audio-text alignment, audio search capabilities

#### 4. Video Modality (Hybrid Fusion)

**Video Encoder**: EgoVLPv2 (Egocentric Video-Language Pre-training v2)
- **Architecture**: TimeSformer-B backbone with cross-modal fusion in backbone
- **Input**: Video frames extracted from video file (MP4, WebM, MOV, AVI)
- **Embedding Dimension**: 768-dim (native)

**Audio Encoder**: CLAP (same as audio modality)
- **Input**: Audio track extracted from video file
- **Embedding Dimension**: 512-dim (native)

**Fusion Strategy**: Concatenation + MLP
```python
# Video-Audio Hybrid Pipeline
video_frames → EgoVLPv2 → video_emb (768-dim)
audio_track → CLAP → audio_emb (512-dim)

# Concatenation
combined = concat([video_emb, audio_emb])  # 768 + 512 = 1280-dim

# Fusion MLP Projection
Video Fusion MLP: 1280 → 1024 → 2048-dim

# Result: Unified 2048-dim embedding in shared space
```

**Video Fusion MLP Architecture:**
```
Input: 1280-dim (EgoVLPv2 768 + CLAP 512)
  ↓
Layer 1: 1280 → 1024 (bottleneck)
  ↓ [LayerNorm, GELU, Dropout(0.1)]
Layer 2: 1024 → 2048 (shared space)
  ↓ [LayerNorm]
Output: 2048-dim (normalized)
```

**Edge Cases:**
- **Silent Videos**: Zero-pad audio embedding (512-dim zeros)
- **Video-Only Processing**: Use EgoVLPv2 only (768-dim) if audio extraction fails
- **Audio Extraction Failure**: Fallback to video-only embedding

### Unified Shared Space

All modalities are projected to a unified **2048-dim shared space** enabling:
- **Cross-modal Search**: Text → Images, Audio, Video (and vice versa)
- **Multimodal Queries**: Combined queries across modalities
- **Unified Vector Store**: Single FAISS/cuVS index for all modalities
- **Semantic Alignment**: Similar content across modalities close in embedding space

### Training Strategy

**MLP Projection Training:**
- **Text-Based Training**: Use text encoders from each model to train MLPs
  - Omura Embed text (768) → BGE text (2048) → Train Image MLP
  - EgoVLPv2 text (RoBERTa-B, 768) → BGE text (2048) → Train Video MLP
  - CLAP text (512) → BGE text (2048) → Train Audio MLP
- **Transfer Learning**: Text-to-text projection transfers to visual/audio embeddings
- **Contrastive Loss**: Cosine similarity loss between projected and target embeddings
- **Video Fusion**: Train with video-audio-text triplets or concatenated text embeddings

### Modality Detection Flow

```mermaid
graph LR
    INPUT[Blob Content<br/>Raw Bytes]
    DETECT[File Type Detection<br/>Magic Bytes]
    
    CHECK{Detected<br/>Type?}
    
    IMAGE[Image:<br/>Omura Embed<br/>768-dim]
    TEXT[Text:<br/>BGE<br/>2048-dim]
    AUDIO[Audio:<br/>CLAP<br/>512-dim]
    VIDEO[Video:<br/>EgoVLPv2 + CLAP<br/>1280 → 2048-dim]
    
    IMAGE_MLP[Image MLP<br/>768 → 2048]
    TEXT_MLP[Text MLP<br/>2048 → 2048]
    AUDIO_MLP[Audio MLP<br/>512 → 2048]
    VIDEO_MLP[Video Fusion MLP<br/>1280 → 2048]
    
    SHARED[Shared Space<br/>2048-dim]
    
    INPUT --> DETECT
    DETECT --> CHECK
    
    CHECK -->|Image| IMAGE
    CHECK -->|Text| TEXT
    CHECK -->|Audio| AUDIO
    CHECK -->|Video| VIDEO
    
    IMAGE --> IMAGE_MLP
    TEXT --> TEXT_MLP
    AUDIO --> AUDIO_MLP
    VIDEO --> VIDEO_MLP
    
    IMAGE_MLP --> SHARED
    TEXT_MLP --> SHARED
    AUDIO_MLP --> SHARED
    VIDEO_MLP --> SHARED
    
    SHARED --> STORE[FAISS/cuVS<br/>Vector Store]
    
    style SHARED fill:#e1ffe1
    style VIDEO fill:#ffe1e1
    style VIDEO_MLP fill:#ffe1e1
```

### Video Processing Pipeline

```mermaid
sequenceDiagram
    participant Indexer as Indexer
    participant VideoProc as Video Processor
    participant EgoVLPv2 as EgoVLPv2 Encoder<br/>(TimeSformer-B)
    participant CLAP as CLAP Encoder<br/>(Audio)
    participant Fusion as Video Fusion MLP
    participant VectorStore as Vector Store
    
    Indexer->>VideoProc: Video file detected<br/>(MP4, WebM, MOV, etc.)
    VideoProc->>VideoProc: Extract video frames<br/>(sample keyframes/temporal)
    VideoProc->>VideoProc: Extract audio track<br/>(from video file)
    
    par Parallel Processing
        VideoProc->>EgoVLPv2: Process video frames<br/>(TimeSformer-B backbone)
        EgoVLPv2->>EgoVLPv2: Video-language fusion<br/>(in backbone)
        EgoVLPv2-->>VideoProc: video_emb (768-dim)
    and
        VideoProc->>CLAP: Process audio track<br/>(CLAP audio encoder)
        CLAP-->>VideoProc: audio_emb (512-dim)
    end
    
    alt Audio extraction fails
        VideoProc->>VideoProc: Zero-pad audio_emb<br/>(512-dim zeros)
    end
    
    VideoProc->>Fusion: Concatenate [video_emb, audio_emb]<br/>1280-dim
    Fusion->>Fusion: Layer 1: 1280 → 1024<br/>(bottleneck, LayerNorm, GELU)
    Fusion->>Fusion: Layer 2: 1024 → 2048<br/>(shared space, LayerNorm)
    Fusion-->>VideoProc: hybrid_emb (2048-dim, normalized)
    
    VideoProc->>VectorStore: Store embedding<br/>blob_id, hybrid_emb, metadata<br/>(modality="video", has_audio=true/false)
    VectorStore-->>Indexer: Indexed successfully
    
    Note over VideoProc,Fusion: Silent videos: zero-pad audio (512-dim)<br/>Video-only: use EgoVLPv2 only (768-dim)
```

---

## Walrus Protocol Integration

### Walrus Protocol Architecture

**Walrus** is a decentralized blob storage protocol built on the Sui blockchain that enables permanent, on-chain blob storage. Key architectural components:

#### 1. On-Chain Storage
- **Blockchain Integration**: Blobs stored directly on Sui blockchain (immutable, permanent)
- **Epoch-Based Lifecycle**: Blobs have `startEpoch` and `endEpoch` defining active period
- **Base64 Identifiers**: Blob IDs are base64-encoded for retrieval (e.g., `DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo`)
- **Owner Management**: Blobs have owners who can delete expired blobs

#### 2. Aggregator Network
- **Distributed Nodes**: Network of aggregator nodes provides HTTP access to blob content
- **Default Aggregator**: `https://walrus-mainnet-aggregator.redundex.com`
- **API Endpoint**: `/v1/blobs/{blob_id}` (HTTP GET)
- **Content Delivery**: Returns raw blob content with appropriate Content-Type headers
- **Load Balancing**: Multiple aggregators ensure high availability

#### 3. Epoch System
- **Sui Epochs**: Periodic epochs on Sui blockchain (time-based progression)
- **Epoch Metadata**: Each blob has `startEpoch` and `endEpoch` timestamps
- **Expiry Logic**: Blob expires when `current_epoch >= end_epoch`
- **Active Condition**: Blob is accessible while `current_epoch < end_epoch`

### Walrus Blob Structure

**Blob Metadata (from Blockberry API):**
```json
{
  "blobIdBase64": "DFBxFKxkkJ23oNMgn_baUcs5HVce_FEGrqJqlruEpRo",
  "size": 123456,
  "owner": "0x1234...",
  "startEpoch": 20,
  "endEpoch": 25,
  "deletable": true,
  "timestamp": "2024-01-15T10:30:00Z",
  "mimeType": "image/png"  // Optional, detected by Omura
}
```

**Key Fields:**
- `blobIdBase64`: Base64-encoded blob identifier (primary key)
- `size`: Blob size in bytes
- `owner`: Sui address of blob owner
- `startEpoch`: Epoch when blob was registered (immutable)
- `endEpoch`: Epoch when blob expires (defines active period)
- `deletable`: Whether blob can be deleted by owner
- `timestamp`: Registration timestamp

### Epoch Tracking & Blob Lifecycle

**Epoch Progression Example:**
```
Epoch 20: Blob registered (startEpoch = 20)
Epoch 21: Blob active (current_epoch = 21, endEpoch = 25) ✓
Epoch 22: Blob active (current_epoch = 22, endEpoch = 25) ✓
Epoch 23: Blob active (current_epoch = 23, endEpoch = 25) ✓
Epoch 24: Blob active (current_epoch = 24, endEpoch = 25) ✓
Epoch 25: Blob expires (current_epoch = 25, endEpoch = 25) ✗
Epoch 26: Blob inactive (current_epoch = 26, endEpoch = 25) ✗
```

**Omura's Epoch Handling:**
1. **Discovery Phase**: Filter blobs where `end_epoch > current_epoch` (active only)
2. **Indexing Phase**: Only index active blobs (prevent wasted compute)
3. **Cleanup Phase**: Periodically remove expired blobs from index via cron job
4. **Update Phase**: Re-evaluate epoch on each indexing cycle

### Walrus Discovery via Blockberry API

**Blockberry API** provides discovery and metadata services for Walrus blobs:

**Endpoints:**
- **Base URL**: `https://api.blockberry.one/walrus-mainnet/v1/blobs`
- **Authentication**: API key via `x-api-key` header
- **Pagination**: Page-based (default: 100 blobs per page)
- **Sorting**: By size (DESC), timestamp (DESC), or epoch
- **Filtering**: Active blobs only (epoch-based)

**API Response Format:**
```json
{
  "content": [
    {
      "blobIdBase64": "...",
      "size": 123456,
      "owner": "0x1234...",
      "startEpoch": 20,
      "endEpoch": 25,
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1000,
  "page": 0,
  "size": 100
}
```

**Omura Discovery Flow:**
1. Query Blockberry API for blob metadata (paginated, sorted by size DESC)
2. Filter active blobs (`end_epoch > current_epoch`)
3. Extract blob IDs and metadata (size, owner, epochs, timestamp)
4. Stream blob IDs to indexer for processing
5. Update cursor for resumption capability

### Walrus Aggregator Integration

**Aggregator Network** provides HTTP access to blob content:

**Default Configuration:**
- **Aggregator URL**: `https://walrus-mainnet-aggregator.redundex.com`
- **Endpoint**: `/v1/blobs/{blob_id}`
- **Method**: HTTP GET
- **Response**: Raw blob content (binary)
- **Timeout**: 60 seconds (configurable)

**Omura Integration:**
- Fetches blob content from aggregator for indexing
- Handles MIME type detection via magic bytes
- Proxies blob content through `/blob/{blob_id}` endpoint
- Supports all Walrus blob types (images, videos, audio, documents, archives)

**Content Access:**
- **Direct Access**: `GET /v1/blobs/{blob_id}` (from aggregator)
- **Proxied Access**: `GET /blob/{blob_id}` (via Omura with MIME detection)
- **Frontend Use**: `<img src="/blob/{blob_id}">` (automatic MIME type handling)

### Blob Expiry & Deletion

**Expiry Behavior:**
- **Automatic Expiry**: Blobs become inaccessible when `current_epoch >= end_epoch`
- **Deletion**: Expired blobs can be deleted by owner (reclaiming storage space)
- **Index Cleanup**: Omura removes expired blobs from index via cron job

**Cleanup Strategy:**
1. **Prevention**: Filter expired blobs during discovery (never index active expired content)
2. **Cron-Based Cleanup**: Periodic cron job removes blobs that expire after indexing (hourly scheduled)
3. **Real-time Epoch Updates**: Re-evaluate blob expiry on epoch changes via WebSocket subscriptions
4. **Database Queries**: Efficient SQL queries in DuckDB with query optimization for expiry checks (`WHERE expiresAt <= current_epoch`)
5. **Distributed Cleanup**: Multi-node cleanup coordination via shared DuckDB state (only leader node executes)
6. **Incremental Cleanup**: Process cleanup in batches to avoid blocking search operations
7. **Automatic Rebuild**: Rebuild FAISS/cuVS index after cleanup completes
8. **Backup Before Cleanup**: Create backup before cleanup operations (safety net)

**Expiry Query Example:**
```sql
-- Query expired blobs for cleanup
SELECT blob_id, expiresAt, indexed_at
FROM blobs
WHERE expiresAt <= (
    SELECT current_epoch FROM cursor WHERE id = 1
)
ORDER BY indexed_at DESC;
```

---

## Indexing Workflow Summary

### Complete Flow

```mermaid
flowchart TD
    START([API Startup])
    
    INIT[Initialize Vector Store<br/>Load from disk if exists]
    LOAD_CURSOR[Load Cursor<br/>from DuckDB]
    
    START_INDEXER[Start Background Indexer<br/>Daemon Thread]
    
    GET_EPOCH[Get Current Epoch<br/>Infer or use MANUAL_EPOCH]
    
    STREAM[Stream Active Blobs<br/>from Blockberry API<br/>page-by-page]
    
    CHECK_CURSOR{Cursor<br/>Resume?}
    SKIP[Skip until<br/>last_blob_id]
    START_FRESH[Start from<br/>beginning]
    
    BATCH[Collect Blobs<br/>Batch size: 100]
    
    PARALLEL[Parallel Processing<br/>4 workers]
    
    FETCH[Fetch Blob Content<br/>from Aggregator]
    DETECT[Detect File Type<br/>Magic Bytes]
    
    FILTER_MODALITY{Modality?}
    IMAGE[Generate Image Embedding<br/>Omura Embed, 768-dim]
    TEXT[Generate Text Embedding<br/>BGE, 2048-dim]
    AUDIO[Generate Audio Embedding<br/>CLAP, 512-dim]
    VIDEO[Generate Video Embedding<br/>EgoVLPv2 (768) + CLAP (512)<br/>→ Fusion MLP → 2048-dim]
    
    ADD[Add to Vector Store<br/>with metadata]
    
    UPDATE_CURSOR[Update Cursor<br/>page, last_blob_id]
    
    CHECK_BATCH{Batch<br/>Count?}
    REBUILD2[Rebuild Index<br/>Every 2 batches]
    SAVE5[Save to Disk<br/>Every 5 batches]
    CONTINUE[Continue]
    
    END_STREAM{End of<br/>Stream?}
    WAIT[Wait 30s<br/>Retry]
    
    ERROR{Error?}
    RETRY[Retry Logic<br/>Max 5 retries]
    SAVE_ERROR[Save Progress<br/>Create Backup]
    
    START --> INIT
    INIT --> LOAD_CURSOR
    LOAD_CURSOR --> START_INDEXER
    START_INDEXER --> GET_EPOCH
    GET_EPOCH --> STREAM
    
    STREAM --> CHECK_CURSOR
    CHECK_CURSOR -->|Resume| SKIP
    CHECK_CURSOR -->|New| START_FRESH
    SKIP --> BATCH
    START_FRESH --> BATCH
    
    BATCH --> PARALLEL
    PARALLEL --> FETCH
    FETCH --> DETECT
    DETECT --> FILTER_MODALITY
    
    FILTER_MODALITY -->|Image| IMAGE
    FILTER_MODALITY -->|Text| TEXT
    FILTER_MODALITY -->|Audio| AUDIO
    FILTER_MODALITY -->|Video| VIDEO
    
    IMAGE --> ADD
    TEXT --> ADD
    AUDIO --> ADD
    VIDEO --> ADD
    
    ADD --> UPDATE_CURSOR
    UPDATE_CURSOR --> CHECK_BATCH
    
    CHECK_BATCH -->|2 batches| REBUILD2
    CHECK_BATCH -->|5 batches| SAVE5
    CHECK_BATCH -->|Other| CONTINUE
    
    REBUILD2 --> CONTINUE
    SAVE5 --> CONTINUE
    CONTINUE --> END_STREAM
    
    END_STREAM -->|More| STREAM
    END_STREAM -->|Complete| WAIT
    WAIT --> STREAM
    
    FETCH -.->|Error| ERROR
    DETECT -.->|Error| ERROR
    IMAGE -.->|Error| ERROR
    TEXT -.->|Error| ERROR
    AUDIO -.->|Error| ERROR
    VIDEO -.->|Error| ERROR
    ADD -.->|Error| ERROR
    
    ERROR -->|Retry| RETRY
    ERROR -->|Max Retries| SAVE_ERROR
    RETRY --> PARALLEL
    SAVE_ERROR --> WAIT
    
    style INIT fill:#e1f5ff
    style GET_EPOCH fill:#ffe1e1
    style DETECT fill:#e1ffe1
    style ADD fill:#fff4e1
```

---

## Key Design Decisions

### 1. **Epoch-Based Filtering**
- **Rationale**: Prevent indexing of expired blobs at discovery phase
- **Benefit**: No wasted compute on expired content
- **Implementation**: Real-time epoch tracking via WebSocket subscriptions to Sui blockchain
- **Epoch Updates**: Automatic re-evaluation on epoch changes with instant filtering

### 2. **Cursor-Based Resumption**
- **Rationale**: Enable crash recovery without re-indexing
- **Benefit**: Efficient incremental updates, progress tracking
- **Implementation**: Stores page number + last processed blob ID

### 3. **Magic Bytes Detection**
- **Rationale**: Accurate file type detection independent of filename/extensions
- **Benefit**: Works with raw blob content, handles misnamed files
- **Fallback**: Heuristic detection for common formats if libmagic unavailable

### 4. **Batch Processing with Periodic Saves**
- **Rationale**: Balance between performance and data safety
- **Benefit**: Efficient indexing with regular persistence
- **Frequency**: Cursor (every batch), Index rebuild (every 2 batches), Save (every 5 batches)

### 5. **Separate Modality Processing with Unified Projection**
- **Rationale**: Different encoders for different content types (optimal quality per modality)
- **Benefit**: Optimal embedding quality per modality + unified cross-modal search
- **Implementation**: Specialized encoders (Omura Embed, BGE, CLAP, EgoVLPv2) with projection MLPs to shared 2048-dim space
- **Cross-Modal Search**: All modalities searchable in unified space (text→image/audio/video, etc.)

---

## Performance Characteristics

### Indexing Performance
- **Throughput**: ~100 blobs per batch (configurable)
- **Parallelism**: 4 worker threads (configurable)
- **Filtering**: Epoch-based filtering happens during discovery (minimal overhead)
- **File Detection**: <1ms per blob (magic bytes scan)

### Storage Efficiency
- **Embeddings**: ~3KB per embedding (768/2048-dim float32)
- **Metadata**: ~200 bytes per blob (stored in DuckDB, compressed)
- **Index Overhead**: FAISS/cuVS index structures (~10-20% of embedding size)

### Reliability
- **Crash Recovery**: Automatic via cursor-based resumption
- **Data Safety**: Periodic saves + automatic backups
- **Error Handling**: Retry logic with exponential backoff

---

### Cleanup Job API Integration

```mermaid
sequenceDiagram
    participant API as FastAPI Server
    participant Cleanup as Cleanup Job
    participant DuckDB as DuckDB
    participant VectorStore as Vector Store
    participant SearchAPI as Search Endpoints
    
    Note over Cleanup: Cron triggers cleanup
    
    Cleanup->>DuckDB: Query expired blobs<br/>SELECT blob_id FROM blobs<br/>WHERE expiresAt <= current_epoch
    
    alt Expired blobs found
        loop For each expired blob
            Cleanup->>VectorStore: Remove embedding (in-memory)
            Cleanup->>DuckDB: DELETE FROM blobs<br/>WHERE blob_id = ?
            Cleanup->>DuckDB: DELETE FROM embeddings<br/>WHERE blob_id = ?
        end
        
        Cleanup->>VectorStore: Rebuild index
        Cleanup->>VectorStore: Save to disk
        
        Note over SearchAPI: Search API automatically<br/>uses updated index
    end
    
    Cleanup->>API: Log cleanup completion
    API->>API: Update /search/stats endpoint<br/>to reflect new count
```

---

## Conclusion

Omura's architecture provides a robust, scalable foundation for multimodal blob search on the Walrus protocol. The combination of epoch-based filtering, cursor-based resumption, and efficient file type detection ensures reliable and performant indexing of active blob content while maintaining data integrity through persistent storage and backup mechanisms.

