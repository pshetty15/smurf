-- GitHub Repository Schema Migration
-- This file extends the existing crawled_pages.sql schema to support GitHub repository data
-- Run this AFTER running crawled_pages.sql

-- Ensure pgvector extension is available (should already be enabled)
create extension if not exists vector;

-- =============================================================================
-- 1. EXTEND EXISTING TABLES FOR GITHUB COMPATIBILITY
-- =============================================================================

-- Add GitHub-specific columns to existing sources table
-- This maintains compatibility with existing web crawling data
alter table sources add column if not exists source_type text default 'web';
alter table sources add column if not exists github_data jsonb default null;

-- Add index for source type filtering
create index if not exists idx_sources_type on sources (source_type);

-- Add GitHub-specific metadata columns to crawled_pages table  
-- These are optional and won't affect existing web crawling data
alter table crawled_pages add column if not exists content_type text default 'documentation';
alter table crawled_pages add column if not exists language text default null;
alter table crawled_pages add column if not exists file_path text default null;
alter table crawled_pages add column if not exists git_info jsonb default null;

-- Add indexes for new columns
create index if not exists idx_crawled_pages_content_type on crawled_pages (content_type);
create index if not exists idx_crawled_pages_language on crawled_pages (language);
create index if not exists idx_crawled_pages_file_path on crawled_pages (file_path);

-- =============================================================================
-- 2. NEW REPOSITORIES TABLE
-- =============================================================================

create table if not exists repositories (
    repo_id text primary key,
    url text not null unique,
    name text not null,
    owner text not null,
    description text,
    last_indexed timestamp with time zone,
    file_count integer default 0,
    primary_language text,
    index_status text default 'pending' check (index_status in ('pending', 'indexing', 'completed', 'failed', 'updating')),
    total_lines integer default 0,
    total_chunks integer default 0,
    languages_distribution jsonb default '{}'::jsonb,
    repository_stats jsonb default '{}'::jsonb,
    indexing_config jsonb default '{}'::jsonb,
    error_message text default null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create indexes for repositories table
create index idx_repositories_owner on repositories (owner);
create index idx_repositories_language on repositories (primary_language);
create index idx_repositories_status on repositories (index_status);
create index idx_repositories_updated on repositories (updated_at);
create index idx_repositories_languages on repositories using gin (languages_distribution);

-- =============================================================================
-- 3. CODE STRUCTURES TABLE  
-- =============================================================================

create table if not exists code_structures (
    id bigserial primary key,
    repo_id text not null,
    file_path text not null,
    element_type text not null check (element_type in ('function', 'class', 'method', 'module', 'import', 'variable', 'constant', 'interface', 'enum', 'decorator')),
    element_name text not null,
    signature text,
    docstring text,
    start_line integer not null,
    end_line integer not null,
    language text not null,
    complexity_score integer default 0,
    dependencies jsonb default '[]'::jsonb,
    called_by jsonb default '[]'::jsonb,
    calls jsonb default '[]'::jsonb,
    imports jsonb default '[]'::jsonb,
    exports jsonb default '[]'::jsonb,
    parameters jsonb default '[]'::jsonb,
    return_type text,
    visibility text default 'public' check (visibility in ('public', 'private', 'protected', 'internal')),
    is_async boolean default false,
    is_static boolean default false,
    decorators jsonb default '[]'::jsonb,
    annotations jsonb default '{}'::jsonb,
    source_code text,
    code_hash text, -- For detecting changes
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Foreign key constraint
    foreign key (repo_id) references repositories(repo_id) on delete cascade,
    
    -- Unique constraint to prevent duplicates
    unique(repo_id, file_path, element_type, element_name, start_line)
);

-- Create indexes for code_structures table
create index idx_code_structures_repo on code_structures (repo_id);
create index idx_code_structures_type on code_structures (element_type);
create index idx_code_structures_language on code_structures (language);
create index idx_code_structures_file on code_structures (file_path);
create index idx_code_structures_name on code_structures (element_name);
create index idx_code_structures_embedding on code_structures using ivfflat (embedding vector_cosine_ops);
create index idx_code_structures_dependencies on code_structures using gin (dependencies);
create index idx_code_structures_calls on code_structures using gin (calls);
create index idx_code_structures_imports on code_structures using gin (imports);
create index idx_code_structures_hash on code_structures (code_hash);

-- =============================================================================
-- 4. REPOSITORY RELATIONSHIPS TABLE
-- =============================================================================

create table if not exists repository_relationships (
    id bigserial primary key,
    repo_id text not null,
    relationship_type text not null check (relationship_type in ('file_dependency', 'function_call', 'class_inheritance', 'module_import', 'interface_implementation')),
    source_element_id bigint,
    target_element_id bigint,
    source_identifier text not null, -- file path, function name, etc.
    target_identifier text not null,
    relationship_data jsonb default '{}'::jsonb,
    strength float default 1.0, -- Relationship strength for graph algorithms
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    foreign key (repo_id) references repositories(repo_id) on delete cascade,
    foreign key (source_element_id) references code_structures(id) on delete cascade,
    foreign key (target_element_id) references code_structures(id) on delete cascade
);

-- Create indexes for relationships
create index idx_repo_relationships_repo on repository_relationships (repo_id);
create index idx_repo_relationships_type on repository_relationships (relationship_type);
create index idx_repo_relationships_source on repository_relationships (source_element_id);
create index idx_repo_relationships_target on repository_relationships (target_element_id);
create index idx_repo_relationships_strength on repository_relationships (strength);

-- =============================================================================
-- 5. CODE PATTERNS TABLE
-- =============================================================================

create table if not exists code_patterns (
    id bigserial primary key,
    repo_id text not null,
    pattern_type text not null check (pattern_type in ('design_pattern', 'anti_pattern', 'code_smell', 'architectural_pattern', 'idiom', 'best_practice')),
    pattern_name text not null,
    description text,
    confidence_score float not null default 0.0 check (confidence_score >= 0.0 and confidence_score <= 1.0),
    file_paths text[] not null,
    element_ids bigint[] default array[]::bigint[],
    pattern_data jsonb default '{}'::jsonb,
    examples jsonb default '[]'::jsonb,
    detected_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    foreign key (repo_id) references repositories(repo_id) on delete cascade
);

-- Create indexes for code patterns
create index idx_code_patterns_repo on code_patterns (repo_id);
create index idx_code_patterns_type on code_patterns (pattern_type);
create index idx_code_patterns_name on code_patterns (pattern_name);
create index idx_code_patterns_confidence on code_patterns (confidence_score);

-- =============================================================================
-- 6. REPOSITORY SNAPSHOTS TABLE (for change tracking)
-- =============================================================================

create table if not exists repository_snapshots (
    id bigserial primary key,
    repo_id text not null,
    snapshot_id text not null unique,
    snapshot_type text default 'full' check (snapshot_type in ('full', 'incremental', 'structure_only')),
    commit_hash text,
    branch_name text default 'main',
    file_count integer default 0,
    total_lines integer default 0,
    structure_hash text, -- Hash of the overall structure for quick comparison
    snapshot_data jsonb not null, -- Compressed representation of the repository state
    metadata jsonb default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    foreign key (repo_id) references repositories(repo_id) on delete cascade
);

-- Create indexes for snapshots
create index idx_repo_snapshots_repo on repository_snapshots (repo_id);
create index idx_repo_snapshots_type on repository_snapshots (snapshot_type);
create index idx_repo_snapshots_created on repository_snapshots (created_at);
create index idx_repo_snapshots_hash on repository_snapshots (structure_hash);

-- =============================================================================
-- 7. ENHANCED SEARCH FUNCTIONS
-- =============================================================================

-- Enhanced match_documents function that works with both web content and GitHub code
create or replace function match_documents_enhanced (
    query_embedding vector(1536),
    match_count int default 10,
    filter jsonb default '{}'::jsonb,
    source_filter text default null,
    content_types text[] default null,
    languages text[] default null,
    include_code boolean default true
) returns table (
    id bigint,
    url varchar,
    chunk_number integer,
    content text,
    metadata jsonb,
    source_id text,
    content_type text,
    language text,
    file_path text,
    similarity float,
    result_type text
)
language plpgsql
as $$
#variable_conflict use_column
begin
    return query
    select
        cp.id,
        cp.url,
        cp.chunk_number,
        cp.content,
        cp.metadata,
        cp.source_id,
        cp.content_type,
        cp.language,
        cp.file_path,
        1 - (cp.embedding <=> query_embedding) as similarity,
        'document'::text as result_type
    from crawled_pages cp
    where cp.metadata @> filter
        and (source_filter is null or cp.source_id = source_filter)
        and (content_types is null or cp.content_type = any(content_types))
        and (languages is null or cp.language = any(languages))
        and cp.embedding is not null
    
    union all
    
    select
        cs.id,
        ('repo://' || cs.repo_id || '/' || cs.file_path)::varchar as url,
        0 as chunk_number,
        coalesce(cs.source_code, cs.signature || E'\n' || coalesce(cs.docstring, '')) as content,
        jsonb_build_object(
            'element_type', cs.element_type,
            'element_name', cs.element_name,
            'signature', cs.signature,
            'start_line', cs.start_line,
            'end_line', cs.end_line,
            'complexity_score', cs.complexity_score,
            'visibility', cs.visibility
        ) as metadata,
        cs.repo_id as source_id,
        'code'::text as content_type,
        cs.language,
        cs.file_path,
        1 - (cs.embedding <=> query_embedding) as similarity,
        'code_structure'::text as result_type
    from code_structures cs
    where include_code = true
        and (source_filter is null or cs.repo_id = source_filter)
        and (languages is null or cs.language = any(languages))
        and cs.embedding is not null
    
    order by similarity desc
    limit match_count;
end;
$$;

-- Function to search specifically within code structures
create or replace function match_code_structures (
    query_embedding vector(1536),
    match_count int default 10,
    repo_filter text default null,
    element_types text[] default null,
    languages text[] default null,
    min_complexity integer default 0,
    max_complexity integer default 999999
) returns table (
    id bigint,
    repo_id text,
    file_path text,
    element_type text,
    element_name text,
    signature text,
    docstring text,
    start_line integer,
    end_line integer,
    language text,
    complexity_score integer,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        cs.id,
        cs.repo_id,
        cs.file_path,
        cs.element_type,
        cs.element_name,
        cs.signature,
        cs.docstring,
        cs.start_line,
        cs.end_line,
        cs.language,
        cs.complexity_score,
        1 - (cs.embedding <=> query_embedding) as similarity
    from code_structures cs
    where (repo_filter is null or cs.repo_id = repo_filter)
        and (element_types is null or cs.element_type = any(element_types))
        and (languages is null or cs.language = any(languages))
        and cs.complexity_score >= min_complexity
        and cs.complexity_score <= max_complexity
        and cs.embedding is not null
    order by cs.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- Function to find similar code structures
create or replace function find_similar_code_structures (
    reference_id bigint,
    match_count int default 10,
    repo_filter text default null,
    same_type_only boolean default true,
    min_similarity float default 0.7
) returns table (
    id bigint,
    repo_id text,
    file_path text,
    element_type text,
    element_name text,
    signature text,
    language text,
    similarity float
)
language plpgsql
as $$
declare
    ref_embedding vector(1536);
    ref_type text;
    ref_language text;
begin
    -- Get reference embedding and metadata
    select embedding, element_type, language
    into ref_embedding, ref_type, ref_language
    from code_structures
    where id = reference_id;
    
    if ref_embedding is null then
        return;
    end if;
    
    return query
    select
        cs.id,
        cs.repo_id,
        cs.file_path,
        cs.element_type,
        cs.element_name,
        cs.signature,
        cs.language,
        1 - (cs.embedding <=> ref_embedding) as similarity
    from code_structures cs
    where cs.id != reference_id
        and (repo_filter is null or cs.repo_id = repo_filter)
        and (not same_type_only or cs.element_type = ref_type)
        and cs.language = ref_language
        and cs.embedding is not null
        and 1 - (cs.embedding <=> ref_embedding) >= min_similarity
    order by cs.embedding <=> ref_embedding
    limit match_count;
end;
$$;

-- Function to get repository dependency graph
create or replace function get_repository_dependencies (
    target_repo_id text,
    relationship_types text[] default array['file_dependency', 'function_call', 'module_import']
) returns table (
    source_file text,
    target_file text,
    relationship_type text,
    source_element text,
    target_element text,
    strength float
)
language plpgsql
as $$
begin
    return query
    select
        rr.source_identifier as source_file,
        rr.target_identifier as target_file,
        rr.relationship_type,
        coalesce(cs1.element_name, rr.source_identifier) as source_element,
        coalesce(cs2.element_name, rr.target_identifier) as target_element,
        rr.strength
    from repository_relationships rr
    left join code_structures cs1 on rr.source_element_id = cs1.id
    left join code_structures cs2 on rr.target_element_id = cs2.id
    where rr.repo_id = target_repo_id
        and (relationship_types is null or rr.relationship_type = any(relationship_types))
    order by rr.strength desc;
end;
$$;

-- =============================================================================
-- 8. ENABLE RLS AND CREATE POLICIES
-- =============================================================================

-- Enable RLS on new tables
alter table repositories enable row level security;
alter table code_structures enable row level security;
alter table repository_relationships enable row level security;
alter table code_patterns enable row level security;
alter table repository_snapshots enable row level security;

-- Create policies for read access
create policy "Allow public read access to repositories"
    on repositories for select to public using (true);

create policy "Allow public read access to code_structures"
    on code_structures for select to public using (true);

create policy "Allow public read access to repository_relationships"
    on repository_relationships for select to public using (true);

create policy "Allow public read access to code_patterns"
    on code_patterns for select to public using (true);

create policy "Allow public read access to repository_snapshots"
    on repository_snapshots for select to public using (true);

-- =============================================================================
-- 9. UTILITY FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Function to update repository statistics
create or replace function update_repository_stats()
returns trigger as $$
begin
    if TG_OP = 'INSERT' or TG_OP = 'UPDATE' then
        update repositories set
            total_chunks = (
                select count(*) from code_structures 
                where repo_id = NEW.repo_id
            ),
            languages_distribution = (
                select jsonb_object_agg(language, count(*))
                from (
                    select language, count(*) 
                    from code_structures 
                    where repo_id = NEW.repo_id 
                    group by language
                ) lang_counts
            ),
            updated_at = now()
        where repo_id = NEW.repo_id;
        return NEW;
    elsif TG_OP = 'DELETE' then
        update repositories set
            total_chunks = (
                select count(*) from code_structures 
                where repo_id = OLD.repo_id
            ),
            updated_at = now()
        where repo_id = OLD.repo_id;
        return OLD;
    end if;
    return null;
end;
$$ language plpgsql;

-- Create trigger to automatically update repository stats
create trigger update_repo_stats_trigger
    after insert or update or delete on code_structures
    for each row execute function update_repository_stats();

-- Function to automatically update the updated_at timestamp
create or replace function update_updated_at_column()
returns trigger as $$
begin
    NEW.updated_at = now();
    return NEW;
end;
$$ language plpgsql;

-- Create triggers for updated_at columns
create trigger update_repositories_updated_at
    before update on repositories
    for each row execute function update_updated_at_column();

create trigger update_code_structures_updated_at
    before update on code_structures
    for each row execute function update_updated_at_column();

-- =============================================================================
-- 10. DATA MIGRATION HELPERS
-- =============================================================================

-- Function to migrate existing web crawl data to new schema
create or replace function migrate_existing_web_data()
returns void as $$
begin
    -- Update sources table to mark existing data as web type
    update sources 
    set source_type = 'web' 
    where source_type is null;
    
    -- Update crawled_pages to mark as documentation type
    update crawled_pages 
    set content_type = 'documentation' 
    where content_type is null;
    
    -- Create repository entries for existing web sources if they look like GitHub URLs
    insert into repositories (repo_id, url, name, owner, index_status, last_indexed)
    select 
        s.source_id,
        'https://' || s.source_id as url,
        split_part(s.source_id, '/', 2) as name,
        split_part(s.source_id, '/', 1) as owner,
        'completed' as index_status,
        s.updated_at as last_indexed
    from sources s
    where s.source_id like 'github.com/%'
        and not exists (select 1 from repositories r where r.repo_id = s.source_id)
    on conflict (repo_id) do nothing;
    
    raise notice 'Migration of existing web data completed';
end;
$$ language plpgsql;

-- =============================================================================
-- 11. PERFORMANCE OPTIMIZATION VIEWS
-- =============================================================================

-- View for repository overview with stats
create or replace view repository_overview as
select 
    r.*,
    s.summary as repository_summary,
    s.total_word_count,
    coalesce(cs_stats.structure_count, 0) as total_structures,
    coalesce(pattern_stats.pattern_count, 0) as total_patterns,
    coalesce(cs_stats.avg_complexity, 0) as avg_complexity
from repositories r
left join sources s on r.repo_id = s.source_id
left join (
    select 
        repo_id,
        count(*) as structure_count,
        avg(complexity_score) as avg_complexity
    from code_structures
    group by repo_id
) cs_stats on r.repo_id = cs_stats.repo_id
left join (
    select 
        repo_id,
        count(*) as pattern_count
    from code_patterns
    group by repo_id
) pattern_stats on r.repo_id = pattern_stats.repo_id;

-- View for code structure summary
create or replace view code_structure_summary as
select 
    cs.*,
    r.name as repo_name,
    r.owner as repo_owner,
    array_length(string_to_array(cs.file_path, '/'), 1) as file_depth,
    case 
        when cs.element_type in ('class', 'interface') then 'type_definition'
        when cs.element_type in ('function', 'method') then 'executable'
        when cs.element_type in ('import', 'module') then 'dependency'
        else 'other'
    end as element_category
from code_structures cs
join repositories r on cs.repo_id = r.repo_id;

-- =============================================================================
-- 12. SAMPLE DATA AND VALIDATION
-- =============================================================================

-- Function to validate schema consistency
create or replace function validate_github_schema()
returns table (
    table_name text,
    constraint_name text,
    status text,
    message text
) language plpgsql as $$
begin
    -- Check if all foreign keys are valid
    return query
    select 
        'code_structures'::text,
        'repo_id_fk'::text,
        case when exists(
            select 1 from code_structures cs
            left join repositories r on cs.repo_id = r.repo_id
            where r.repo_id is null
        ) then 'INVALID' else 'VALID' end,
        'Foreign key constraint between code_structures and repositories'::text;
    
    -- Check embedding consistency
    return query
    select 
        'code_structures'::text,
        'embedding_check'::text,
        case when exists(
            select 1 from code_structures 
            where embedding is null and element_type in ('function', 'class', 'method')
        ) then 'WARNING' else 'VALID' end,
        'Important code elements should have embeddings'::text;
    
    -- Check repository status consistency
    return query
    select 
        'repositories'::text,
        'status_consistency'::text,
        case when exists(
            select 1 from repositories r
            left join code_structures cs on r.repo_id = cs.repo_id
            where r.index_status = 'completed' and cs.id is null
        ) then 'WARNING' else 'VALID' end,
        'Completed repositories should have code structures'::text;
end;
$$;

-- =============================================================================
-- FINAL NOTES
-- =============================================================================

-- This migration extends the existing schema to support GitHub repositories while
-- maintaining full compatibility with existing web crawling functionality.
-- 
-- Key features added:
-- 1. Repository tracking and metadata
-- 2. Detailed code structure analysis
-- 3. Relationship mapping between code elements
-- 4. Pattern recognition capabilities
-- 5. Change tracking through snapshots
-- 6. Enhanced search functions for code
-- 7. Automatic statistics and maintenance
-- 
-- To apply this migration:
-- 1. First ensure crawled_pages.sql has been run
-- 2. Run this file: psql -f github_repos.sql
-- 3. Optionally run: select migrate_existing_web_data();
-- 4. Validate: select * from validate_github_schema();

-- Create a comment for documentation
comment on schema public is 'Extended schema supporting both web crawling and GitHub repository analysis';

-- Log successful migration
do $$
begin
    raise notice 'GitHub repository schema migration completed successfully';
    raise notice 'Tables created: repositories, code_structures, repository_relationships, code_patterns, repository_snapshots';
    raise notice 'Enhanced functions: match_documents_enhanced, match_code_structures, find_similar_code_structures';
    raise notice 'Views created: repository_overview, code_structure_summary';
end;
$$;