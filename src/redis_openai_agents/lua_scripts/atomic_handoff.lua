-- atomic_handoff.lua
-- Atomically perform agent handoff with distributed locking
--
-- KEYS[1] = session key (e.g., "session:abc123")
-- KEYS[2] = lock key (e.g., "session:abc123:handoff_lock")
-- ARGV[1] = from_agent
-- ARGV[2] = to_agent
-- ARGV[3] = context JSON
-- ARGV[4] = lock_ttl (seconds, default 30)
--
-- Returns: "OK" on success, or table with err="HANDOFF_IN_PROGRESS" if locked

local session_key = KEYS[1]
local lock_key = KEYS[2]
local from_agent = ARGV[1]
local to_agent = ARGV[2]
local context_json = ARGV[3]
local lock_ttl = tonumber(ARGV[4]) or 30

-- Get current timestamp
local now = redis.call('TIME')
local timestamp = tonumber(now[1]) + tonumber(now[2]) / 1000000

-- 1. Try to acquire lock (NX = only if not exists)
local lock_acquired = redis.call('SET', lock_key, timestamp, 'NX', 'EX', lock_ttl)

if not lock_acquired then
    -- Lock already held - another handoff in progress
    return cjson.encode({err = "HANDOFF_IN_PROGRESS"})
end

-- 2. Update current agent
redis.call('JSON.SET', session_key, '$.metadata.current_agent', '"' .. to_agent .. '"')

-- 3. Add to_agent to agents_used if not already present
local agents_used = redis.call('JSON.GET', session_key, '$.metadata.agents_used')
if agents_used then
    local agents = cjson.decode(agents_used)
    if agents and agents[1] then
        local found = false
        for _, agent in ipairs(agents[1]) do
            if agent == to_agent then
                found = true
                break
            end
        end
        if not found then
            redis.call('JSON.ARRAPPEND', session_key, '$.metadata.agents_used', '"' .. to_agent .. '"')
        end
    end
end

-- 4. Store handoff context
local context = cjson.decode(context_json)
context.from_agent = from_agent
context.to_agent = to_agent
context.timestamp = timestamp
redis.call('JSON.SET', session_key, '$.handoff_context', cjson.encode(context))

-- 5. Update session timestamp
redis.call('JSON.SET', session_key, '$.updated_at', timestamp)

-- 6. Keep lock for a short time to prevent rapid-fire concurrent handoffs
-- The lock will expire after lock_ttl seconds (default 30)
-- This prevents multiple handoffs from being initiated simultaneously

return "OK"
