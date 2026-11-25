-- NAYRA Module 10: Object Location Memory SQLite Schema

CREATE TABLE IF NOT EXISTS object_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_label TEXT NOT NULL,              -- Raw class name (e.g., 'cell phone', 'remote', 'bottle')
    canonical_name TEXT NOT NULL,            -- Standardized query key (e.g., 'phone', 'remote', 'bottle')
    hindi_alias TEXT,                        -- Hindi display label (e.g., 'फ़ोन', 'रिमोट')
    timestamp REAL NOT NULL,                 -- Unix epoch timestamp
    iso_time TEXT NOT NULL,                  -- ISO-8601 formatted string
    confidence REAL NOT NULL,                -- Detection confidence [0.0 - 1.0]
    track_id INTEGER,                        -- ByteTrack ID
    spatial_relation TEXT,                   -- 'on', 'near', 'next to', 'in'
    anchor_object TEXT,                      -- Supporting/reference object (e.g., 'dining table', 'couch', 'bed')
    horizontal_zone TEXT,                    -- 'LEFT', 'CENTER', 'RIGHT'
    vertical_zone TEXT DEFAULT NULL,         -- 'UPPER', 'MIDDLE', 'LOWER'
    frame_width INTEGER DEFAULT 640,         -- Image width for spatial coordinate reference
    frame_height INTEGER DEFAULT 480,        -- Image height for spatial coordinate reference
    room_label TEXT DEFAULT NULL,            -- Assigned room if reliably established
    bbox_json TEXT NOT NULL,                 -- '[x1, y1, x2, y2]'
    snapshot_path TEXT DEFAULT NULL,         -- Path to stored cropped image (optional)
    is_latest INTEGER DEFAULT 1              -- 1 if most recent observation for this object
);

CREATE INDEX IF NOT EXISTS idx_obs_canonical ON object_observations (canonical_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_obs_latest ON object_observations (canonical_name, is_latest);
CREATE INDEX IF NOT EXISTS idx_obs_track ON object_observations (track_id);
CREATE INDEX IF NOT EXISTS idx_obs_time ON object_observations (timestamp DESC);

CREATE TABLE IF NOT EXISTS room_anchors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_name TEXT UNIQUE NOT NULL,          -- 'bedroom', 'living room', 'kitchen'
    signature_objects TEXT NOT NULL,         -- JSON array: '["bed"]', '["refrigerator"]'
    confirmed_by_user INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_query TEXT NOT NULL,
    matched_object TEXT,
    retrieved_observation_id INTEGER,
    response_text TEXT NOT NULL,
    query_timestamp REAL NOT NULL,
    FOREIGN KEY(retrieved_observation_id) REFERENCES object_observations(id)
);
