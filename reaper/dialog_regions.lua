-- Show Region Marker — background script for REAPER
-- Polls /tmp/reaper_state.txt for state changes and creates colored regions.
-- Backend writes "dialog", "ad", or "ident" to the file.
-- Run via Actions → Run ReaScript before or during recording.

local STATE_FILE = "/tmp/reaper_state.txt"

local COLORS = {
  dialog = reaper.ColorToNative(50, 180, 50) + 0x1000000,   -- green
  ad     = reaper.ColorToNative(200, 80, 80) + 0x1000000,   -- red
  ident  = reaper.ColorToNative(80, 120, 200) + 0x1000000,  -- blue
}
local LABELS = {
  dialog = "DIALOG",
  ad     = "AD",
  ident  = "IDENT",
}

local counts = { dialog = 0, ad = 0, ident = 0 }
local current_type = nil    -- which region type is currently open
local current_start = 0
local last_pos = 0          -- last known transport position (while running)
local last_state = ""
local transport_active = false

local function log(msg)
  reaper.ShowConsoleMsg("[Regions] " .. msg .. "\n")
end

local function is_playing_or_recording()
  local state = reaper.GetPlayState()
  return state > 0 and state ~= 2
end

local function open_region(rtype)
  if current_type then return end
  current_type = rtype
  current_start = reaper.GetPlayPosition()
  log("OPEN " .. rtype .. " at " .. string.format("%.2f", current_start))
end

local function close_region(pos_override)
  if not current_type then return end
  local pos = pos_override or reaper.GetPlayPosition()
  local len = pos - current_start
  local rtype = current_type
  current_type = nil
  log("CLOSE " .. rtype .. " at " .. string.format("%.2f", pos) .. " (len=" .. string.format("%.2f", len) .. ")")
  if len > 0.1 then
    counts[rtype] = counts[rtype] + 1
    local name = LABELS[rtype] .. " " .. counts[rtype]
    reaper.AddProjectMarker2(0, true, current_start, pos, name, -1, COLORS[rtype])
    log("  -> Created '" .. name .. "'")
  else
    log("  -> Skipped (too short)")
  end
end

local function poll()
  if not transport_active then
    if is_playing_or_recording() then
      transport_active = true
      log("Transport started at " .. string.format("%.2f", reaper.GetPlayPosition()))
      local f = io.open(STATE_FILE, "r")
      if f then
        last_state = f:read("*l") or "dialog"
        f:close()
      else
        last_state = "dialog"
      end
      log("Initial state: '" .. last_state .. "'")
      open_region(last_state)
    end
    reaper.defer(poll)
    return
  end

  -- Track position while transport is running
  last_pos = reaper.GetPlayPosition()

  -- Detect transport stop (recording ended) — use last known good position
  if not is_playing_or_recording() then
    log("Transport stopped at last known pos " .. string.format("%.2f", last_pos))
    close_region(last_pos)
    transport_active = false
    reaper.defer(poll)
    return
  end

  local f = io.open(STATE_FILE, "r")
  if f then
    local state = f:read("*l") or "dialog"
    f:close()

    if state ~= last_state then
      log("State change: '" .. last_state .. "' -> '" .. state .. "'")
      close_region()
      open_region(state)
      last_state = state
    end
  end

  reaper.defer(poll)
end

log("Script loaded — waiting for transport to start...")

reaper.atexit(function()
  log("Script stopping (atexit)")
  close_region()
  local total = counts.dialog + counts.ad + counts.ident
  log("Done. " .. total .. " regions (" .. counts.dialog .. " dialog, " .. counts.ad .. " ad, " .. counts.ident .. " ident)")
end)

poll()
