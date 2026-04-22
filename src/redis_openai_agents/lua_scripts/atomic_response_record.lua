-- atomic_response_record.lua
-- Atomically record response: update cache, append messages, update metrics
--
-- KEYS[1] = session key (e.g., "session:abc123")
-- KEYS[2] = cache key (e.g., "cache:semantic")
-- ARGV[1] = query_hash
-- ARGV[2] = response text
-- ARGV[3] = user_message JSON
-- ARGV[4] = assistant_message JSON
-- ARGV[5] = latency_ms
-- ARGV[6] = input_tokens
-- ARGV[7] = output_tokens
-- ARGV[8] = cache_ttl (0 = no expiration)
-- ARGV[9] = max_messages (0 = no limit)
--
-- Returns: "OK" on success

local session_key = KEYS[1]
local cache_key = KEYS[2]
local query_hash = ARGV[1]
local response = ARGV[2]
local user_message_json = ARGV[3]
local assistant_message_json = ARGV[4]
local latency_ms = tonumber(ARGV[5]) or 0
local input_tokens = tonumber(ARGV[6]) or 0
local output_tokens = tonumber(ARGV[7]) or 0
local cache_ttl = tonumber(ARGV[8]) or 0
local max_messages = tonumber(ARGV[9]) or 0

-- Get current timestamp
local now = redis.call('TIME')
local timestamp = tonumber(now[1]) + tonumber(now[2]) / 1000000

-- 1. Store in cache (hash field)
local cache_entry = cjson.encode({
    response = response,
    timestamp = timestamp,
    latency_ms = latency_ms,
    input_tokens = input_tokens,
    output_tokens = output_tokens
})
redis.call('HSET', cache_key, query_hash, cache_entry)

-- Set cache TTL if specified
if cache_ttl > 0 then
    redis.call('EXPIRE', cache_key, cache_ttl)
end

-- 2. Append user message to session
redis.call('JSON.ARRAPPEND', session_key, '$.messages', user_message_json)
redis.call('JSON.NUMINCRBY', session_key, '$.metadata.message_count', 1)

-- 3. Append assistant message to session
redis.call('JSON.ARRAPPEND', session_key, '$.messages', assistant_message_json)
redis.call('JSON.NUMINCRBY', session_key, '$.metadata.message_count', 1)

-- 4. Update token totals
local total_tokens = input_tokens + output_tokens
if total_tokens > 0 then
    redis.call('JSON.NUMINCRBY', session_key, '$.metadata.total_tokens', total_tokens)
end

-- 5. Update session timestamp
redis.call('JSON.SET', session_key, '$.updated_at', timestamp)

-- 6. Trim messages if max_messages specified
if max_messages > 0 then
    local current_len = redis.call('JSON.ARRLEN', session_key, '$.messages')
    if current_len and current_len[1] then
        local length = current_len[1]
        if length > max_messages then
            local to_remove = length - max_messages
            for i = 1, to_remove do
                redis.call('JSON.ARRPOP', session_key, '$.messages', 0)
            end
        end
    end
end

return "OK"
