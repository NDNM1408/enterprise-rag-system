CREATE TYPE "DocumentStatus" AS ENUM ('Created', 'Processing', 'Succeed', 'Failed');
CREATE TYPE "ChunkStatus" AS ENUM ('Processing', 'Succeed', 'Failed');

CREATE TABLE "knowledge_base" (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    description TEXT,
    tenant_id VARCHAR,
    created_by VARCHAR,
    embed_id VARCHAR,
    parser_id VARCHAR,
    parser_config JSON,
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW()
);

CREATE TABLE "document" (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    kb_id VARCHAR REFERENCES "knowledge_base"(id) ON DELETE CASCADE,
    cmetadata JSON,
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW(),
    status "DocumentStatus" NOT NULL
);

CREATE TABLE "chunk" (
    id VARCHAR PRIMARY KEY,
    content TEXT,
    document_id VARCHAR REFERENCES "document"(id) ON DELETE CASCADE,
    doc_name VARCHAR,
    status "ChunkStatus" NOT NULL,
    create_time TIMESTAMP NOT NULL DEFAULT NOW(),
    update_time TIMESTAMP DEFAULT NOW()
);
