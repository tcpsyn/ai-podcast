-- Bleep Selection — censor a time range on the selected track(s)
--
-- Replaces the audio inside the time selection with a 1kHz tone rather than
-- muting it. Muted regions read as silence to strip_silence_dialog.lua
-- (SILENCE_DB -30, thresholds 5-6s), so a muted bleep over a slowly-read phone
-- number can get stripped and shift everything after it out of sync. A tone at
-- TONE_GAIN_DB is nowhere near -30, so the silence pass leaves it alone.
--
-- Usage: make a time selection over the digits, select the track, run.

---------------------------------------------------------------------------
-- SETTINGS
---------------------------------------------------------------------------
local TONE_GAIN_DB = 0.0    -- adjust bleep level; file is generated at -15 dBFS
local FADE_MS      = 4.0    -- fade in/out on the tone, prevents clicks
local TONE_FILE    = "bleep_1khz.wav"  -- 60s 1kHz sine @48k, sits next to this script

---------------------------------------------------------------------------

local EPS = 1e-9

local function script_dir()
  local src = debug.getinfo(1, "S").source
  return src:match("@?(.*[/\\])") or ""
end

local function items_on(track)
  local t = {}
  for i = 0, reaper.CountTrackMediaItems(track) - 1 do
    t[#t + 1] = reaper.GetTrackMediaItem(track, i)
  end
  return t
end

local function item_bounds(item)
  local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  return pos, pos + reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
end

-- Split every item crossing either edge so the range is cleanly separable
local function split_at_edges(track, sel_start, sel_end)
  for _, item in ipairs(items_on(track)) do
    local pos, fin = item_bounds(item)
    if fin > sel_start + EPS and pos < sel_end - EPS then
      local right = item
      if pos < sel_start - EPS then
        right = reaper.SplitMediaItem(item, sel_start)
      end
      if right then
        local rpos, rfin = item_bounds(right)
        if rfin > sel_end + EPS and rpos < sel_end - EPS then
          reaper.SplitMediaItem(right, sel_end)
        end
      end
    end
  end
end

local function delete_inside(track, sel_start, sel_end)
  local removed = 0
  for _, item in ipairs(items_on(track)) do
    local pos, fin = item_bounds(item)
    if pos >= sel_start - EPS and fin <= sel_end + EPS then
      reaper.DeleteTrackMediaItem(track, item)
      removed = removed + 1
    end
  end
  return removed
end

-- Build the item directly rather than via InsertMedia(): InsertMedia behaves
-- like the user-facing "insert media file" action — it obeys ripple editing
-- (shifting other tracks), can spawn a new track, and moves the edit cursor.
-- AddMediaItemToTrack touches nothing but this track.
local function insert_tone(track, sel_start, sel_len, tone_source)
  local item = reaper.AddMediaItemToTrack(track)
  if not item then return nil end
  local take = reaper.AddTakeToMediaItem(item)
  if not take then return nil end
  reaper.SetMediaItemTake_Source(take, tone_source)

  reaper.SetMediaItemInfo_Value(item, "D_POSITION", sel_start)
  reaper.SetMediaItemInfo_Value(item, "D_LENGTH", sel_len)
  reaper.SetMediaItemInfo_Value(item, "B_LOOPSRC", 0)
  reaper.SetMediaItemInfo_Value(item, "D_VOL", 10 ^ (TONE_GAIN_DB / 20))

  local fade = math.min(FADE_MS / 1000, sel_len / 2)
  reaper.SetMediaItemInfo_Value(item, "D_FADEINLEN", fade)
  reaper.SetMediaItemInfo_Value(item, "D_FADEOUTLEN", fade)
  return item
end

---------------------------------------------------------------------------

local function main()
  local sel_start, sel_end = reaper.GetSet_LoopTimeRange(false, false, 0, 0, false)
  local sel_len = sel_end - sel_start
  if sel_len <= 0 then
    reaper.ShowMessageBox("Make a time selection over the audio to bleep.", "Bleep Selection", 0)
    return
  end

  local n_tracks = reaper.CountSelectedTracks(0)
  if n_tracks == 0 then
    reaper.ShowMessageBox("Select the track to bleep.", "Bleep Selection", 0)
    return
  end

  local tone_path = script_dir() .. TONE_FILE
  local f = io.open(tone_path, "rb")
  if not f then
    reaper.ShowMessageBox("Tone file not found:\n" .. tone_path, "Bleep Selection", 0)
    return
  end
  f:close()

  local tone_source = reaper.PCM_Source_CreateFromFile(tone_path)
  if not tone_source then
    reaper.ShowMessageBox("Could not load tone file:\n" .. tone_path, "Bleep Selection", 0)
    return
  end

  local targets = {}
  for i = 0, n_tracks - 1 do
    targets[#targets + 1] = reaper.GetSelectedTrack(0, i)
  end

  reaper.Undo_BeginBlock()
  reaper.PreventUIRefresh(1)

  -- Ripple editing would shift unrelated items (and other tracks) when items
  -- are removed. Force it off for the duration, restore the user's mode after.
  local ripple_per_track = reaper.GetToggleCommandStateEx(0, 40310) == 1
  local ripple_all = reaper.GetToggleCommandStateEx(0, 40311) == 1
  if ripple_per_track or ripple_all then
    reaper.Main_OnCommand(40309, 0)   -- ripple editing off
  end

  local bleeped = 0
  for _, track in ipairs(targets) do
    split_at_edges(track, sel_start, sel_end)
    delete_inside(track, sel_start, sel_end)
    if insert_tone(track, sel_start, sel_len, tone_source) then
      bleeped = bleeped + 1
    end
  end

  if ripple_all then
    reaper.Main_OnCommand(40311, 0)
  elseif ripple_per_track then
    reaper.Main_OnCommand(40310, 0)
  end

  reaper.PreventUIRefresh(-1)
  reaper.UpdateArrange()
  reaper.Undo_EndBlock(string.format("Bleep %.2fs on %d track(s)", sel_len, bleeped), -1)
end

main()
