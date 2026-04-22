-- atomic_message_append.lua
-- Atomically append a message to session and update metadata
--
-- KEYS[1] = session key (e.g., "session:abc123")
-- ARGV[1] = message JSON
-- ARGV[2] = max_messages (0 = no limit)
-- ARGV[3] = ttl (0 = no expiration)
--
-- Returns: new message count

local session_key = KEYS[1]
local message_json = ARGV[1]
local max_messages = tonumber(ARGV[2]) or 0
local ttl = tonumber(ARGV[3]) or 0

-- Append message to messages array
redis.call('JSON.ARRAPPEND', session_key, '$.messages', message_json)

-- Increment message count atomically
local new_count = redis.call('JSON.NUMINCRBY', session_key, '$.metadata.message_count', 1)

-- Update timestamp
local now = redis.call('TIME')
local timestamp = tonumber(now[1]) + tonumber(now[2]) / 1000000
redis.call('JSON.SET', session_key, '$.updated_at', timestamp)

-- Trim to max_messages if specified (sliding window)
if max_messages > 0 then
    local current_len = redis.call('JSON.ARRLEN', session_key, '$.messages')
    if current_len and current_len[1] then
        local length = current_len[1]
        if length > max_messages then
            -- Remove oldest messages
            local to_remove = length - max_messages
            for i = 1, to_remove do
                redis.call('JSON.ARRPOP', session_key, '$.messages', 0)
            end
        end
    end
end

-- Set/refresh TTL if specified
if ttl > 0 then
    redis.call('EXPIRE', session_key, ttl)
end

return new_count
