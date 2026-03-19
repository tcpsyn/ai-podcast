-- Post-Production Script for REAPER
-- Phase 1: Strip long silences from DIALOG regions (all tracks except music)
-- Phase 2: Normalize AD/IDENT/music volume to match dialog
-- Phase 3: Trim music to length of longest voice track with fade-out
-- Phase 4: Mute music during AD/IDENT regions with fade in/out

---------------------------------------------------------------------------
-- SETTINGS
---------------------------------------------------------------------------
local SILENCE_DB       = -30    -- dBFS — anything below this is "silence"
local MIN_SILENCE_SEC  = 6.0   -- same-speaker gaps: only remove silences longer than this
local MAX_SILENCE_SEC  = 999   -- no practical limit (IDENT/AD regions protect real breaks)
local MIN_SILENCE_TRANSITION_SEC = 5.0 -- cross-speaker gaps: threshold for caller TTS latency
local MIN_SILENCE_DEVON_SEC = 3.0 -- Devon gaps: interjections are prerendered (~2-3s gaps), conversational TTS is 6s+
local DEVON_TRACK = 2 -- 1-indexed: Devon track number
local MIN_VOICE_SEC    = 0.3   -- ignore non-silent bursts shorter than this (filters transients)
local KEEP_PAD_SEC     = 0.5   -- leave this much silence on each side of a cut
local BLOCK_SEC        = 0.1   -- analysis block size (100ms)
local SAMPLE_RATE      = 48000
local CHECK_TRACKS     = {1, 2, 3, 4} -- 1-indexed: Host, Devon, AI Caller, Live Caller
local IDENTS_TRACK     = 6     -- 1-indexed: Idents track
local ADS_TRACK        = 7     -- 1-indexed: Ads track
local MUSIC_TRACK      = 8     -- 1-indexed: Music track
local MUSIC_FADE_SEC   = 2.0   -- fade duration for music in/out around ads/idents
local YIELD_INTERVAL   = 200   -- yield to REAPER every N blocks (~20s of audio)
---------------------------------------------------------------------------

local BLOCK_SAMPLES = math.floor(SAMPLE_RATE * BLOCK_SEC)
local THRESHOLD = 10 ^ (SILENCE_DB / 20)
local MIN_VOICE_BLOCKS = math.ceil(MIN_VOICE_SEC / BLOCK_SEC)
local function log(msg)
  reaper.ShowConsoleMsg("[PostProd] " .. msg .. "\n")
end

---------------------------------------------------------------------------
-- Progress window (gfx)
---------------------------------------------------------------------------

local progress_phase = ""
local progress_pct = 0
local progress_detail = ""

local function progress_init()
  gfx.init("Post-Production", 420, 60)
  gfx.setfont(1, "Arial", 14)
end

local function progress_draw()
  if gfx.getchar() < 0 then return false end
  gfx.set(0.12, 0.12, 0.12)
  gfx.rect(0, 0, 420, 60, true)
  -- Label
  gfx.set(1, 1, 1)
  gfx.x = 10; gfx.y = 8
  gfx.drawstr(progress_phase)
  gfx.x = 300; gfx.y = 8
  gfx.drawstr(progress_detail)
  -- Bar background
  gfx.set(0.25, 0.25, 0.25)
  gfx.rect(10, 32, 400, 18, true)
  -- Bar fill
  gfx.set(0.2, 0.7, 0.3)
  local fill = math.min(math.floor(400 * progress_pct), 400)
  if fill > 0 then gfx.rect(10, 32, fill, 18, true) end
  gfx.update()
  return true
end

local function progress_close()
  gfx.quit()
end

---------------------------------------------------------------------------
-- Region helpers
---------------------------------------------------------------------------

local function get_regions_by_type(type_pattern)
  local regions = {}
  local _, num_markers, num_regions = reaper.CountProjectMarkers(0)
  local total = num_markers + num_regions
  for i = 0, total - 1 do
    local retval, is_region, pos, rgnend, name, idx = reaper.EnumProjectMarkers(i)
    if is_region and name and name:match(type_pattern) then
      table.insert(regions, {start_pos = pos, end_pos = rgnend, name = name})
    end
  end
  table.sort(regions, function(a, b) return a.start_pos < b.start_pos end)
  return regions
end

local function merge_regions(regions)
  if #regions <= 1 then return regions end
  table.sort(regions, function(a, b) return a.start_pos < b.start_pos end)
  local merged = {{start_pos = regions[1].start_pos, end_pos = regions[1].end_pos, name = "MERGED 1"}}
  for i = 2, #regions do
    local prev = merged[#merged]
    if regions[i].start_pos <= prev.end_pos then
      prev.end_pos = math.max(prev.end_pos, regions[i].end_pos)
    else
      table.insert(merged, {start_pos = regions[i].start_pos, end_pos = regions[i].end_pos, name = "MERGED " .. (#merged + 1)})
    end
  end
  return merged
end

local function shift_regions(removals)
  local _, num_markers, num_regions = reaper.CountProjectMarkers(0)
  local total_markers = num_markers + num_regions

  local markers = {}
  for i = 0, total_markers - 1 do
    local retval, is_region, pos, rgnend, name, idx, color = reaper.EnumProjectMarkers3(0, i)
    if retval then
      table.insert(markers, {is_region=is_region, pos=pos, rgnend=rgnend, name=name, idx=idx, color=color})
    end
  end

  for _, m in ipairs(markers) do
    local pos_shift = 0
    for _, r in ipairs(removals) do
      if r.end_pos <= m.pos then
        pos_shift = pos_shift + (r.end_pos - r.start_pos)
      elseif r.start_pos < m.pos then
        pos_shift = pos_shift + (m.pos - r.start_pos)
      end
    end
    m.new_pos = m.pos - pos_shift

    if m.is_region then
      local end_shift = 0
      for _, r in ipairs(removals) do
        if r.end_pos <= m.rgnend then
          end_shift = end_shift + (r.end_pos - r.start_pos)
        elseif r.start_pos < m.rgnend then
          end_shift = end_shift + (m.rgnend - r.start_pos)
        end
      end
      m.new_end = m.rgnend - end_shift
    end
  end

  for _, m in ipairs(markers) do
    if m.is_region then
      reaper.SetProjectMarker3(0, m.idx, true, m.new_pos, m.new_end, m.name, m.color)
    else
      reaper.SetProjectMarker3(0, m.idx, false, m.new_pos, 0, m.name, m.color)
    end
  end
end

local function find_item_at(track, pos)
  for i = 0, reaper.CountTrackMediaItems(track) - 1 do
    local item = reaper.GetTrackMediaItem(track, i)
    local item_start = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
    local item_len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
    if pos >= item_start and pos < item_start + item_len then
      return item
    end
  end
  return nil
end

---------------------------------------------------------------------------
-- Phase 1: Silence detection and removal
---------------------------------------------------------------------------

-- Read audio directly from WAV files (bypasses REAPER accessor — immune to undo issues)
local function parse_wav_header(filepath)
  local f = io.open(filepath, "rb")
  if not f then return nil end
  local riff = f:read(4)
  if riff ~= "RIFF" then f:close(); return nil end
  f:read(4)  -- file size
  if f:read(4) ~= "WAVE" then f:close(); return nil end
  local fmt_info = nil
  while true do
    local id = f:read(4)
    if not id then f:close(); return nil end
    local size = string.unpack("<I4", f:read(4))
    if id == "fmt " then
      local audio_fmt = string.unpack("<I2", f:read(2))
      local channels = string.unpack("<I2", f:read(2))
      local sr = string.unpack("<I4", f:read(4))
      f:read(4)  -- byte rate
      f:read(2)  -- block align
      local bps = string.unpack("<I2", f:read(2))
      if size > 16 then f:read(size - 16) end
      fmt_info = {audio_fmt = audio_fmt, channels = channels, sample_rate = sr, bps = bps}
    elseif id == "data" then
      if not fmt_info then f:close(); return nil end
      local data_offset = f:seek()
      f:close()
      fmt_info.data_offset = data_offset
      fmt_info.data_size = size
      fmt_info.filepath = filepath
      fmt_info.bytes_per_sample = fmt_info.bps / 8
      fmt_info.frame_size = fmt_info.channels * fmt_info.bytes_per_sample
      return fmt_info
    else
      f:read(size)
    end
  end
end

local function get_track_audio(track_idx_1based)
  local track = reaper.GetTrack(0, track_idx_1based - 1)
  if not track or reaper.CountTrackMediaItems(track) == 0 then return nil end

  local segments = {}
  for i = 0, reaper.CountTrackMediaItems(track) - 1 do
    local item = reaper.GetTrackMediaItem(track, i)
    local take = reaper.GetActiveTake(item)
    if take then
      local source = reaper.GetMediaItemTake_Source(take)
      local filepath = reaper.GetMediaSourceFileName(source)
      local wav = parse_wav_header(filepath)
      if wav then
        local item_pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
        local item_len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
        local take_offset = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
        local fh = io.open(filepath, "rb")
        if fh then
          table.insert(segments, {
            fh = fh,
            wav = wav,
            item_pos = item_pos,
            item_end = item_pos + item_len,
            take_offset = take_offset,
          })
        end
      else
        log("  WARNING: Could not parse WAV header for: " .. filepath)
      end
    end
  end

  if #segments == 0 then return nil end

  -- Sort by position so binary-style lookup is possible
  table.sort(segments, function(a, b) return a.item_pos < b.item_pos end)

  return {
    segments = segments,
    item_pos = segments[1].item_pos,
    item_end = segments[#segments].item_end,
  }
end

local function destroy_track_audio(ta)
  for _, seg in ipairs(ta.segments) do
    if seg.fh then seg.fh:close(); seg.fh = nil end
  end
end

local function read_block_peak_rms_segment(seg, project_time)
  local source_time = project_time - seg.item_pos + seg.take_offset
  if source_time < 0 then return 0, 0 end

  local wav = seg.wav
  local sample_offset = math.floor(source_time * wav.sample_rate)
  local byte_offset = wav.data_offset + sample_offset * wav.frame_size
  local bytes_needed = BLOCK_SAMPLES * wav.frame_size

  if byte_offset + bytes_needed > wav.data_offset + wav.data_size then
    return 0, 0
  end

  seg.fh:seek("set", byte_offset)
  local raw = seg.fh:read(bytes_needed)
  if not raw or #raw < bytes_needed then return 0, 0 end

  local peak = 0
  local sum_sq = 0
  local bps = wav.bytes_per_sample

  for i = 0, BLOCK_SAMPLES - 1 do
    local offset = i * wav.frame_size
    local v = 0
    if wav.audio_fmt == 3 then
      v = string.unpack("<f", raw, offset + 1)
    elseif bps == 3 then
      local b1, b2, b3 = string.byte(raw, offset + 1, offset + 3)
      local val = b1 + b2 * 256 + b3 * 65536
      if val >= 8388608 then val = val - 16777216 end
      v = val / 8388608.0
    elseif bps == 2 then
      v = string.unpack("<i2", raw, offset + 1) / 32768.0
    elseif bps == 4 and wav.audio_fmt == 1 then
      v = string.unpack("<i4", raw, offset + 1) / 2147483648.0
    end

    sum_sq = sum_sq + v * v
    local av = math.abs(v)
    if av > peak then peak = av end
  end

  return peak, sum_sq
end

local function read_block_peak_rms(ta, project_time)
  -- Find the segment that contains this project time
  for _, seg in ipairs(ta.segments) do
    if project_time >= seg.item_pos and project_time < seg.item_end then
      return read_block_peak_rms_segment(seg, project_time)
    end
  end
  return 0, 0
end

-- find_loudest_track: returns 1-based index of the loudest track at a given time, or 0 if silent
-- Uses RMS (not peak) for speaker identification — ambient mic noise has high peaks but low RMS
local function find_loudest_track(track_audios, project_time)
  local best_peak = 0
  local best_rms = 0
  local best_idx = 0
  for i, ta in ipairs(track_audios) do
    local peak, sum_sq = read_block_peak_rms(ta, project_time)
    if peak > best_peak then best_peak = peak end
    local rms = math.sqrt(sum_sq / BLOCK_SAMPLES)
    if rms > best_rms then
      best_rms = rms
      best_idx = i
    end
  end
  if best_peak < THRESHOLD then return 0 end
  return best_idx
end

-- find_silences: detects silences and accumulates RMS data
-- Tracks which track was active before/after each silence to distinguish
-- speaker transitions (short threshold) from same-speaker pauses (long threshold).
-- Yields periodically via coroutine for UI responsiveness
-- progress_fn(t): called before each yield with current position
local function find_silences(region, track_audios, rms_acc, progress_fn)
  local silences = {}
  local in_silence = false
  local silence_start = 0
  local track_before_silence = 0
  local voice_run = 0
  local voice_run_track = 0
  local last_active_track = 0
  local t = region.start_pos
  local total_blocks = 0
  local silent_blocks = 0
  local yield_count = 0

  while t < region.end_pos do
    local best_peak = 0
    local best_rms = 0
    local best_sum = 0
    local best_track = 0
    for i, ta in ipairs(track_audios) do
      local peak, sum_sq = read_block_peak_rms(ta, t)
      if peak > best_peak then best_peak = peak end
      -- Use RMS for speaker identification (sustained energy, not transient peaks)
      -- Host mic ambient noise has high peaks but low RMS; TTS speech has high RMS
      local rms = math.sqrt(sum_sq / BLOCK_SAMPLES)
      if rms > best_rms then
        best_rms = rms
        best_sum = sum_sq
        best_track = i
      end
    end

    local all_silent = best_peak < THRESHOLD
    total_blocks = total_blocks + 1
    if all_silent then silent_blocks = silent_blocks + 1 end

    if not all_silent then
      last_active_track = best_track
      if rms_acc then
        rms_acc.sum_sq = rms_acc.sum_sq + best_sum
        rms_acc.count = rms_acc.count + BLOCK_SAMPLES
      end
    end

    if in_silence then
      if all_silent then
        voice_run = 0
        voice_run_track = 0
      else
        if voice_run == 0 then voice_run_track = best_track end
        voice_run = voice_run + 1
        if voice_run >= MIN_VOICE_BLOCKS then
          local voice_start = t - (voice_run - 1) * BLOCK_SEC
          local dur = voice_start - silence_start
          local track_after = voice_run_track
          local is_transition = track_before_silence ~= 0 and track_after ~= 0 and track_before_silence ~= track_after
          local devon_involved = track_before_silence == DEVON_TRACK or track_after == DEVON_TRACK
          local threshold = devon_involved and MIN_SILENCE_DEVON_SEC
                         or (is_transition and MIN_SILENCE_TRANSITION_SEC or MIN_SILENCE_SEC)

          if dur >= threshold and dur <= MAX_SILENCE_SEC then
            table.insert(silences, {
              start_pos = silence_start, end_pos = voice_start, duration = dur,
              is_transition = is_transition,
            })
          end
          in_silence = false
          voice_run = 0
          voice_run_track = 0
        end
      end
    else
      if all_silent then
        in_silence = true
        silence_start = t
        track_before_silence = last_active_track
        voice_run = 0
        voice_run_track = 0
      end
    end

    t = t + BLOCK_SEC

    -- Yield periodically so REAPER stays responsive
    yield_count = yield_count + 1
    if yield_count >= YIELD_INTERVAL then
      yield_count = 0
      if progress_fn then progress_fn(t) end
      coroutine.yield()
    end
  end

  if in_silence then
    local dur = region.end_pos - silence_start
    if dur >= MIN_SILENCE_SEC and dur <= MAX_SILENCE_SEC then
      table.insert(silences, {start_pos = silence_start, end_pos = region.end_pos, duration = dur})
    end
  end

  return silences, total_blocks, silent_blocks
end

local function phase1_strip_silence(dialog_regions)
  dialog_regions = merge_regions(dialog_regions)
  log("Phase 1: " .. #dialog_regions .. " merged DIALOG region(s)")

  local track_audios = {}
  local tracks_loaded = 0
  for _, tidx in ipairs(CHECK_TRACKS) do
    local ta = get_track_audio(tidx)
    if ta then
      table.insert(track_audios, ta)
      tracks_loaded = tracks_loaded + 1
      local first_wav = ta.segments[1].wav
      local fmt = first_wav.audio_fmt == 3 and "float" or (first_wav.bps .. "bit")
      log("  Track " .. tidx .. ": " .. #ta.segments .. " item(s), " .. fmt .. " " .. first_wav.sample_rate .. "Hz (pos=" .. string.format("%.1f", ta.item_pos) .. " end=" .. string.format("%.1f", ta.item_end) .. ")")
    else
      log("  WARNING: Track " .. tidx .. " has no audio items — silence detection will NOT check this track")
    end
  end

  if tracks_loaded == 0 then
    log("Phase 1: No audio found on voice tracks — skipping")
    return false, 0
  end

  if tracks_loaded < #CHECK_TRACKS then
    log("  *** Only " .. tracks_loaded .. "/" .. #CHECK_TRACKS .. " voice tracks have audio — silence may be over-detected ***")
  end

  -- Load AD/IDENT regions so we can protect them from silence removal
  local protected_regions = {}
  for _, r in ipairs(get_regions_by_type("^AD%s+%d+$")) do table.insert(protected_regions, r) end
  for _, r in ipairs(get_regions_by_type("^IDENT%s+%d+$")) do table.insert(protected_regions, r) end
  table.sort(protected_regions, function(a, b) return a.start_pos < b.start_pos end)
  if #protected_regions > 0 then
    log("  Protecting " .. #protected_regions .. " AD/IDENT region(s) from silence removal:")
    for _, pr in ipairs(protected_regions) do
      log("    " .. pr.name .. " at " .. string.format("%.1f", pr.start_pos) .. "-" .. string.format("%.1f", pr.end_pos) .. "s")
    end
  end

  log("Phase 1: Analyzing using " .. tracks_loaded .. "/" .. #CHECK_TRACKS .. " voice tracks")
  log("  threshold=" .. SILENCE_DB .. "dB, min_silence=" .. MIN_SILENCE_SEC .. "s (same-speaker), " .. MIN_SILENCE_TRANSITION_SEC .. "s (transition), pad=" .. KEEP_PAD_SEC .. "s")

  -- Calculate total duration for progress tracking
  local total_duration = 0
  for _, rgn in ipairs(dialog_regions) do
    total_duration = total_duration + (rgn.end_pos - rgn.start_pos)
  end
  local processed_duration = 0

  local rms_acc = {sum_sq = 0, count = 0}

  local removals = {}
  local total_blocks = 0
  local silent_blocks = 0
  for ri, rgn in ipairs(dialog_regions) do
    local rgn_dur = rgn.end_pos - rgn.start_pos

    local function update_progress(t)
      local rgn_progress = (t - rgn.start_pos) / rgn_dur
      progress_pct = (processed_duration + rgn_progress * rgn_dur) / total_duration
      progress_phase = "Phase 1: Scanning"
      progress_detail = string.format("Region %d/%d", ri, #dialog_regions)
    end

    local silences, rgn_total, rgn_silent = find_silences(rgn, track_audios, rms_acc, update_progress)
    processed_duration = processed_duration + rgn_dur
    total_blocks = total_blocks + rgn_total
    silent_blocks = silent_blocks + rgn_silent
    log("  " .. rgn.name .. ": " .. rgn_total .. " blocks, " .. rgn_silent .. " silent (" .. string.format("%.0f", rgn_silent/math.max(rgn_total,1)*100) .. "%)")
    for _, s in ipairs(silences) do
      local rm_start = s.start_pos + KEEP_PAD_SEC
      local rm_end = s.end_pos - KEEP_PAD_SEC
      if rm_end > rm_start + 0.05 then
        -- Check if this silence overlaps with any AD/IDENT region
        local protected = false
        for _, pr in ipairs(protected_regions) do
          if rm_start < pr.end_pos and rm_end > pr.start_pos then
            protected = true
            log("    SKIP " .. string.format("%.1f", rm_end - rm_start) .. "s at " .. string.format("%.1f", s.start_pos) .. "-" .. string.format("%.1f", s.end_pos) .. " (overlaps " .. pr.name .. ")")
            break
          end
        end
        -- Preserve the very first silence (music intro before host starts talking)
        if not protected and ri == 1 and #removals == 0 and s.start_pos <= rgn.start_pos + 1.0 then
          protected = true
          log("    KEEP " .. string.format("%.1f", rm_end - rm_start) .. "s at " .. string.format("%.1f", s.start_pos) .. "-" .. string.format("%.1f", s.end_pos) .. " (music intro)")
        end
        if not protected then
          table.insert(removals, {start_pos = rm_start, end_pos = rm_end})
          local tag = s.is_transition and " [transition]" or ""
          log("    remove " .. string.format("%.1f", rm_end - rm_start) .. "s at " .. string.format("%.1f", s.start_pos) .. "-" .. string.format("%.1f", s.end_pos) .. tag)
        end
      end
    end
  end

  for _, ta in ipairs(track_audios) do
    destroy_track_audio(ta)
  end

  log("Phase 1: Total " .. total_blocks .. " blocks, " .. silent_blocks .. " silent (" .. string.format("%.0f", silent_blocks/math.max(total_blocks,1)*100) .. "%)")

  local dialog_rms_db = nil
  if rms_acc.count > 0 then
    local rms = math.sqrt(rms_acc.sum_sq / rms_acc.count)
    if rms > 0 then dialog_rms_db = 20 * math.log(rms, 10) end
  end

  if #removals == 0 then
    log("Phase 1: No long silences found")
    return true, dialog_rms_db
  end

  local total_removed = 0
  for _, r in ipairs(removals) do
    total_removed = total_removed + (r.end_pos - r.start_pos)
  end

  local msg = string.format(
    "Phase 1: Found %d silence(s) totaling %.1fs to remove.\n\nProceed?",
    #removals, total_removed
  )
  if reaper.ShowMessageBox(msg, "Strip Silence", 1) ~= 1 then return false end

  -- Modification phase — prevent UI refresh for performance, but yield for progress
  progress_phase = "Phase 1: Removing"
  reaper.PreventUIRefresh(1)

  for i = #removals, 1, -1 do
    local r = removals[i]
    local remove_len = r.end_pos - r.start_pos

    for t = 0, reaper.CountTracks(0) - 1 do
      if (t + 1) == MUSIC_TRACK then goto next_track end
      local track = reaper.GetTrack(0, t)

      local item = find_item_at(track, r.start_pos)
      if item then
        local right = reaper.SplitMediaItem(item, r.start_pos)
        if right then
          reaper.SplitMediaItem(right, r.end_pos)
          reaper.DeleteTrackMediaItem(track, right)
        end
      end

      for j = 0, reaper.CountTrackMediaItems(track) - 1 do
        local shift_item = reaper.GetTrackMediaItem(track, j)
        local pos = reaper.GetMediaItemInfo_Value(shift_item, "D_POSITION")
        if pos >= r.start_pos then
          reaper.SetMediaItemInfo_Value(shift_item, "D_POSITION", pos - remove_len)
        end
      end

      ::next_track::
    end

    -- Yield every 5 removals to update progress
    if i % 5 == 0 then
      progress_pct = (#removals - i) / #removals
      progress_detail = string.format("%d/%d cuts", #removals - i, #removals)
      reaper.PreventUIRefresh(-1)
      coroutine.yield()
      reaper.PreventUIRefresh(1)
    end
  end

  reaper.PreventUIRefresh(-1)

  shift_regions(removals)
  log("Phase 1: Removed " .. #removals .. " silence(s), " .. string.format("%.1f", total_removed) .. "s total")
  return true, dialog_rms_db
end

---------------------------------------------------------------------------
-- Phase 2: Normalize AD/IDENT volume to match dialog
---------------------------------------------------------------------------

local function normalize_track_items(track_idx, target_db, label)
  -- Normalize all items on a track that have audible content.
  -- Uses direct WAV reading (not audio accessor) so it works after Phase 1 splits.
  local track = reaper.GetTrack(0, track_idx - 1)
  if not track or reaper.CountTrackMediaItems(track) == 0 then return end

  local ta = get_track_audio(track_idx)
  if not ta then
    log("  " .. label .. ": no audio found")
    return
  end

  local adjusted = 0
  for i = 0, reaper.CountTrackMediaItems(track) - 1 do
    local item = reaper.GetTrackMediaItem(track, i)
    local item_pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
    local item_len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
    local item_end = item_pos + item_len

    -- Measure RMS of audible content in this item
    local sum_sq = 0
    local count = 0
    local t = item_pos
    while t < item_end do
      local peak, s_sq = read_block_peak_rms(ta, t)
      if peak >= THRESHOLD then
        sum_sq = sum_sq + s_sq
        count = count + BLOCK_SAMPLES
      end
      t = t + BLOCK_SEC
    end

    if count > 0 then
      local item_rms = math.sqrt(sum_sq / count)
      if item_rms > 0 then
        local item_db = 20 * math.log(item_rms, 10)
        local gain_db = target_db - item_db
        -- Only adjust if the difference is significant (> 1dB)
        if math.abs(gain_db) > 1.0 then
          local gain_linear = 10 ^ (gain_db / 20)
          local current_vol = reaper.GetMediaItemInfo_Value(item, "D_VOL")
          reaper.SetMediaItemInfo_Value(item, "D_VOL", current_vol * gain_linear)
          log("  " .. label .. " item at " .. string.format("%.0f", item_pos) .. "s: " .. string.format("%+.1f", gain_db) .. "dB")
          adjusted = adjusted + 1
        end
      end
    end
  end

  destroy_track_audio(ta)
  if adjusted == 0 then
    log("  " .. label .. ": no adjustments needed")
  end
end

local function normalize_music_track(dialog_regions, target_db)
  local track = reaper.GetTrack(0, MUSIC_TRACK - 1)
  if not track or reaper.CountTrackMediaItems(track) == 0 then return end

  local sum_sq = 0
  local count = 0

  for _, rgn in ipairs(dialog_regions) do
    for i = 0, reaper.CountTrackMediaItems(track) - 1 do
      local item = reaper.GetTrackMediaItem(track, i)
      local take = reaper.GetActiveTake(item)
      if not take then goto next_item end

      local item_pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
      local item_len = reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
      local item_end = item_pos + item_len
      local take_offset = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")

      local mstart = math.max(item_pos, rgn.start_pos)
      local mend = math.min(item_end, rgn.end_pos)
      if mstart >= mend then goto next_item end

      local accessor = reaper.CreateTakeAudioAccessor(take)
      local t = mstart
      while t < mend do
        local source_time = t - item_pos + take_offset
        local buf = reaper.new_array(BLOCK_SAMPLES)
        reaper.GetAudioAccessorSamples(accessor, SAMPLE_RATE, 1, source_time, BLOCK_SAMPLES, buf)
        local peak = 0
        local block_sum = 0
        for j = 1, BLOCK_SAMPLES do
          local v = buf[j]
          block_sum = block_sum + v * v
          local av = math.abs(v)
          if av > peak then peak = av end
        end
        if peak >= THRESHOLD then
          sum_sq = sum_sq + block_sum
          count = count + BLOCK_SAMPLES
        end
        t = t + BLOCK_SEC
      end
      reaper.DestroyAudioAccessor(accessor)

      ::next_item::
    end
  end

  if count == 0 then
    log("  Music: no audio detected — skipping")
    return
  end

  local music_rms = math.sqrt(sum_sq / count)
  if music_rms > 0 then
    local music_db = 20 * math.log(music_rms, 10)
    local gain_db = target_db - music_db
    local gain_linear = 10 ^ (gain_db / 20)

    for i = 0, reaper.CountTrackMediaItems(track) - 1 do
      local item = reaper.GetTrackMediaItem(track, i)
      local current_vol = reaper.GetMediaItemInfo_Value(item, "D_VOL")
      reaper.SetMediaItemInfo_Value(item, "D_VOL", current_vol * gain_linear)
    end
    log("  Music: " .. string.format("%+.1f", gain_db) .. "dB adjustment")
  end
end

local function phase2_normalize(dialog_regions, ad_regions, ident_regions, dialog_rms_db)
  progress_phase = "Phase 2: Normalizing"
  progress_pct = 0
  progress_detail = ""
  coroutine.yield()

  if not dialog_rms_db then
    log("Phase 2: Could not measure dialog loudness — skipping")
    return
  end

  log("Phase 2: Dialog RMS = " .. string.format("%.1f", dialog_rms_db) .. " dBFS")

  -- Ads/idents are pre-compressed dense audio, so they sound louder than dialog
  -- at the same RMS. Target a few dB below dialog to match perceived loudness.
  local AD_IDENT_OFFSET_DB = -4
  local ad_ident_target = dialog_rms_db + AD_IDENT_OFFSET_DB
  log("Phase 2: AD/IDENT target = " .. string.format("%.1f", ad_ident_target) .. " dBFS (" .. AD_IDENT_OFFSET_DB .. "dB offset from dialog)")

  progress_detail = "Ads"
  coroutine.yield()
  log("Phase 2: Normalizing ads track...")
  normalize_track_items(ADS_TRACK, ad_ident_target, "Ads")

  progress_detail = "Idents"
  progress_pct = 0.33
  coroutine.yield()
  log("Phase 2: Normalizing idents track...")
  normalize_track_items(IDENTS_TRACK, ad_ident_target, "Idents")

  progress_detail = "Music"
  progress_pct = 0.66
  coroutine.yield()
  log("Phase 2: Normalizing music track...")
  normalize_music_track(dialog_regions, dialog_rms_db)
  progress_pct = 1.0
end

---------------------------------------------------------------------------
-- Phase 3: Trim music to voice length
-- Phase 4: Mute music during AD/IDENT regions with fades
---------------------------------------------------------------------------

local function phase3_trim_music()
  progress_phase = "Phase 3: Trimming music"
  progress_pct = 0
  progress_detail = ""
  coroutine.yield()

  local music_track = reaper.GetTrack(0, MUSIC_TRACK - 1)
  if not music_track then return end

  -- Music lead-in: ensure audible music plays before first voice.
  -- Strategy: skip the silent intro in the music WAV (adjust take offset),
  -- then nudge all non-music tracks forward by MUSIC_LEAD_SEC so music plays first.
  local MUSIC_LEAD_SEC = 3.0

  -- Find where music becomes audible in the source WAV
  local music_audible_offset = nil
  local music_ta = get_track_audio(MUSIC_TRACK)
  if music_ta then
    local t = music_ta.item_pos
    while t < music_ta.item_end do
      local peak, _ = read_block_peak_rms(music_ta, t)
      if peak >= THRESHOLD then
        music_audible_offset = t - music_ta.item_pos  -- offset into the WAV
        break
      end
      t = t + BLOCK_SEC
    end
    destroy_track_audio(music_ta)
  end

  if false then  -- Music lead-in disabled — intro silence is preserved instead
    -- Skip the silent intro: set take offset so audible music starts at position 0
    local first_music = reaper.GetTrackMediaItem(music_track, 0)
    if first_music then
      local take = reaper.GetActiveTake(first_music)
      if take then
        local current_offset = reaper.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS")
        reaper.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS", current_offset + music_audible_offset)
        -- Trim item length to account for skipped intro
        local item_len = reaper.GetMediaItemInfo_Value(first_music, "D_LENGTH")
        reaper.SetMediaItemInfo_Value(first_music, "D_LENGTH", item_len - music_audible_offset)
        log("Phase 3: Skipped " .. string.format("%.1f", music_audible_offset) .. "s of silent music intro")
      end
    end

    -- Nudge all non-music tracks forward by MUSIC_LEAD_SEC
    log("Phase 3: Nudging non-music tracks forward by " .. MUSIC_LEAD_SEC .. "s for music lead-in")
    for t = 0, reaper.CountTracks(0) - 1 do
      if (t + 1) == MUSIC_TRACK then goto skip_music end
      local track = reaper.GetTrack(0, t)
      for i = 0, reaper.CountTrackMediaItems(track) - 1 do
        local item = reaper.GetTrackMediaItem(track, i)
        local pos = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
        reaper.SetMediaItemInfo_Value(item, "D_POSITION", pos + MUSIC_LEAD_SEC)
      end
      ::skip_music::
    end

    -- Shift markers/regions forward too
    local markers_to_update = {}
    local _, num_markers, num_regions = reaper.CountProjectMarkers(0)
    for i = 0, num_markers + num_regions - 1 do
      local retval, is_region, pos, rgnend, name, idx, color = reaper.EnumProjectMarkers3(0, i)
      if retval then
        table.insert(markers_to_update, {is_region=is_region, pos=pos, rgnend=rgnend, name=name, idx=idx, color=color})
      end
    end
    for _, m in ipairs(markers_to_update) do
      if m.is_region then
        reaper.SetProjectMarker3(0, m.idx, true, m.pos + MUSIC_LEAD_SEC, m.rgnend + MUSIC_LEAD_SEC, m.name, m.color)
      else
        reaper.SetProjectMarker3(0, m.idx, false, m.pos + MUSIC_LEAD_SEC, 0, m.name, m.color)
      end
    end
  else
    log("Phase 3: No silent music intro detected — skipping lead-in adjustment")
  end

  local last_end = 0
  for _, tidx in ipairs(CHECK_TRACKS) do
    local tr = reaper.GetTrack(0, tidx - 1)
    if tr then
      local n = reaper.CountTrackMediaItems(tr)
      if n > 0 then
        local item = reaper.GetTrackMediaItem(tr, n - 1)
        local item_end = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
                       + reaper.GetMediaItemInfo_Value(item, "D_LENGTH")
        if item_end > last_end then last_end = item_end end
      end
    end
  end
  if last_end == 0 then return end

  local item = find_item_at(music_track, last_end - 0.01)
  if not item then
    local n = reaper.CountTrackMediaItems(music_track)
    if n > 0 then
      item = reaper.GetTrackMediaItem(music_track, n - 1)
    end
  end
  if not item then
    log("Phase 3: No music item to trim")
    return
  end

  local item_start = reaper.GetMediaItemInfo_Value(item, "D_POSITION")
  local item_end = item_start + reaper.GetMediaItemInfo_Value(item, "D_LENGTH")

  if last_end < item_end then
    reaper.SetMediaItemInfo_Value(item, "D_LENGTH", last_end - item_start)
    reaper.SetMediaItemInfo_Value(item, "D_FADEOUTLEN", MUSIC_FADE_SEC)
    log("Phase 3: Trimmed music at " .. string.format("%.1f", last_end) .. "s with " .. MUSIC_FADE_SEC .. "s fade-out")

    local i = reaper.CountTrackMediaItems(music_track) - 1
    while i >= 0 do
      local check = reaper.GetTrackMediaItem(music_track, i)
      local check_start = reaper.GetMediaItemInfo_Value(check, "D_POSITION")
      if check_start >= last_end then
        reaper.DeleteTrackMediaItem(music_track, check)
      end
      i = i - 1
    end
  else
    log("Phase 3: Music already ends before last voice audio — adding fade-out")
    reaper.SetMediaItemInfo_Value(item, "D_FADEOUTLEN", MUSIC_FADE_SEC)
  end
  progress_pct = 1.0
end

local function phase4_music_fades(ad_ident_regions)
  progress_phase = "Phase 4: Music fades"
  progress_pct = 0
  progress_detail = ""
  coroutine.yield()

  local music_track = reaper.GetTrack(0, MUSIC_TRACK - 1)
  if not music_track or reaper.CountTrackMediaItems(music_track) == 0 then
    log("Phase 4: No music track/items found — skipping")
    return
  end

  log("Phase 4: Processing " .. #ad_ident_regions .. " AD/IDENT region(s)...")

  for ri, rgn in ipairs(ad_ident_regions) do
    local fade_point = rgn.start_pos - MUSIC_FADE_SEC
    local item = find_item_at(music_track, math.max(fade_point, 0))
    if not item then
      item = find_item_at(music_track, rgn.start_pos)
    end
    if not item then
      log("  " .. rgn.name .. ": no music item found — skipping")
      goto continue
    end

    local item_start = reaper.GetMediaItemInfo_Value(item, "D_POSITION")

    local split_pos = math.max(fade_point, item_start + 0.01)
    local mid = reaper.SplitMediaItem(item, split_pos)
    if mid then
      reaper.SetMediaItemInfo_Value(item, "D_FADEOUTLEN", MUSIC_FADE_SEC)
      local after = reaper.SplitMediaItem(mid, rgn.end_pos)
      reaper.SetMediaItemInfo_Value(mid, "B_MUTE", 1)
      if after then
        reaper.SetMediaItemInfo_Value(after, "D_FADEINLEN", MUSIC_FADE_SEC)
      end
      log("  " .. rgn.name .. ": muted music, fade out/in (" .. MUSIC_FADE_SEC .. "s)")
    end

    progress_pct = ri / #ad_ident_regions
    progress_detail = string.format("%d/%d", ri, #ad_ident_regions)

    ::continue::
  end
end

---------------------------------------------------------------------------
-- Main (coroutine-based for UI responsiveness)
---------------------------------------------------------------------------

local function do_work()
  local dialog_regions = get_regions_by_type("^DIALOG%s+%d+$")
  if #dialog_regions == 0 then
    reaper.ShowMessageBox("No DIALOG regions found.", "Post-Production", 0)
    return
  end

  reaper.Undo_BeginBlock()

  -- Phase 1: Strip silence (analysis yields for progress, removal uses PreventUIRefresh)
  local ok, dialog_rms_db = phase1_strip_silence(dialog_regions)
  if not ok then
    reaper.Undo_EndBlock("Post-production: cancelled", -1)
    log("Cancelled.")
    return
  end

  -- Re-read regions after ripple edits
  dialog_regions = get_regions_by_type("^DIALOG%s+%d+$")
  local ad_regions = get_regions_by_type("^AD%s+%d+$")
  local ident_regions = get_regions_by_type("^IDENT%s+%d+$")
  local ad_ident_regions = {}
  for _, r in ipairs(ad_regions) do table.insert(ad_ident_regions, r) end
  for _, r in ipairs(ident_regions) do table.insert(ad_ident_regions, r) end
  table.sort(ad_ident_regions, function(a, b) return a.start_pos < b.start_pos end)

  reaper.PreventUIRefresh(1)

  -- Phase 2: Normalize
  if #ad_regions > 0 or #ident_regions > 0 then
    phase2_normalize(dialog_regions, ad_regions, ident_regions, dialog_rms_db)
  else
    log("Phase 2: No AD/IDENT regions found — skipping")
  end

  -- Phase 3: Trim music
  phase3_trim_music()

  -- Phase 4: Music fades
  if #ad_ident_regions > 0 then
    phase4_music_fades(ad_ident_regions)
  else
    log("Phase 4: No AD/IDENT regions found — skipping")
  end

  -- Set loop/time selection: start 0.5s before audible music, end at last item
  local loop_start = 0
  local music_ta = get_track_audio(MUSIC_TRACK)
  if music_ta then
    local t = music_ta.item_pos
    while t < music_ta.item_end do
      local peak, _ = read_block_peak_rms(music_ta, t)
      if peak >= THRESHOLD then
        loop_start = math.max(0, t - 0.5)
        break
      end
      t = t + BLOCK_SEC
    end
    destroy_track_audio(music_ta)
  end

  local project_end = 0
  for t = 0, reaper.CountTracks(0) - 1 do
    local track = reaper.GetTrack(0, t)
    local n = reaper.CountTrackMediaItems(track)
    if n > 0 then
      local last_item = reaper.GetTrackMediaItem(track, n - 1)
      local item_end = reaper.GetMediaItemInfo_Value(last_item, "D_POSITION")
                     + reaper.GetMediaItemInfo_Value(last_item, "D_LENGTH")
      if item_end > project_end then project_end = item_end end
    end
  end
  if project_end > 0 then
    reaper.GetSet_LoopTimeRange(true, true, loop_start, project_end, false)
    reaper.GetSet_LoopTimeRange(true, false, loop_start, project_end, false)
    log("Loop range set: " .. string.format("%.1f", loop_start) .. " to " .. string.format("%.1f", project_end) .. "s (" .. string.format("%.1f", (project_end - loop_start) / 60) .. " min)")
  end

  reaper.PreventUIRefresh(-1)
  reaper.Undo_EndBlock("Post-production: strip silence + music fades", -1)
  reaper.UpdateArrange()
  log("All phases complete!")
end

-- Coroutine runner with progress window
local work_co

local function work_loop()
  if not work_co or coroutine.status(work_co) == "dead" then
    progress_phase = "Done!"
    progress_pct = 1.0
    progress_detail = ""
    progress_draw()
    progress_close()
    return
  end

  progress_draw()

  local ok, err = coroutine.resume(work_co)
  if not ok then
    progress_close()
    log("ERROR: " .. tostring(err))
    reaper.PreventUIRefresh(-1)
    reaper.Undo_EndBlock("Post-production: error", -1)
    return
  end

  if coroutine.status(work_co) ~= "dead" then
    reaper.defer(work_loop)
  else
    progress_phase = "Done!"
    progress_pct = 1.0
    progress_detail = ""
    progress_draw()
    progress_close()
  end
end

progress_init()
work_co = coroutine.create(do_work)
reaper.defer(work_loop)
