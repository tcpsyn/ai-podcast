-- Stub harness for reaper/bleep_selection.lua
-- Models tracks/items in memory and asserts the split/delete/insert behaviour.

local SCRIPT = (debug.getinfo(1,"S").source:match("@?(.*[/\\])") or "") .. "bleep_selection.lua"

local function new_item(pos, len, tag)
  return { pos = pos, len = len, tag = tag or "audio", vol = 1, fin_ = 0, fout_ = 0 }
end

local W  -- world

local function make_reaper()
  return {
    CountTrackMediaItems = function(tr) return #tr.items end,
    GetTrackMediaItem = function(tr, i) return tr.items[i + 1] end,
    GetMediaItemInfo_Value = function(it, k)
      if k == "D_POSITION" then return it.pos end
      if k == "D_LENGTH" then return it.len end
      return 0
    end,
    SetMediaItemInfo_Value = function(it, k, v)
      if k == "D_POSITION" then it.pos = v
      elseif k == "D_LENGTH" then it.len = v
      elseif k == "D_VOL" then it.vol = v
      elseif k == "D_FADEINLEN" then it.fin_ = v
      elseif k == "D_FADEOUTLEN" then it.fout_ = v end
    end,
    SplitMediaItem = function(it, at)
      local tr
      for _, t in ipairs(W.tracks) do
        for idx, x in ipairs(t.items) do if x == it then tr = t; it_idx = idx end end
      end
      if not tr then return nil end
      if at <= it.pos or at >= it.pos + it.len then return nil end
      local right = new_item(at, it.pos + it.len - at, it.tag)
      it.len = at - it.pos
      local pos_in = 0
      for idx, x in ipairs(tr.items) do if x == it then pos_in = idx end end
      table.insert(tr.items, pos_in + 1, right)
      W.splits = W.splits + 1
      return right
    end,
    DeleteTrackMediaItem = function(tr, it)
      for idx, x in ipairs(tr.items) do
        if x == it then table.remove(tr.items, idx); W.deletes = W.deletes + 1; return true end
      end
      return false
    end,
    InsertMedia = function() W.forbidden["InsertMedia"] = true; return 1 end,
    GetSelectedMediaItem = function(_, _) return W.last_inserted end,
    SetOnlyTrackSelected = function(tr) W.forbidden["SetOnlyTrackSelected"] = true end,
    SetTrackSelected = function() W.forbidden["SetTrackSelected"] = true end,
    SetEditCurPos = function(p) W.forbidden["SetEditCurPos"] = true end,
    PCM_Source_CreateFromFile = function(path) W.src_path = path; return { src = path } end,
    AddMediaItemToTrack = function(tr)
      local it = new_item(0, 0, "tone"); table.insert(tr.items, it)
      W.last_inserted = it; it.owner = tr; return it
    end,
    AddTakeToMediaItem = function(it) it.take = { item = it }; return it.take end,
    SetMediaItemTake_Source = function(take, src) take.src = src end,
    GetToggleCommandStateEx = function(_, cmd) return W.ripple[cmd] and 1 or 0 end,
    Main_OnCommand = function(cmd) W.commands[#W.commands + 1] = cmd end,
    GetSet_LoopTimeRange = function() return W.sel_start, W.sel_end end,
    CountSelectedTracks = function() return #W.sel_tracks end,
    GetSelectedTrack = function(_, i) return W.sel_tracks[i + 1] end,
    Undo_BeginBlock = function() end,
    Undo_EndBlock = function(desc) W.undo = desc end,
    PreventUIRefresh = function() end,
    UpdateArrange = function() end,
    ShowMessageBox = function(msg) W.msg = msg end,
  }
end

local function run(setup)
  W = { tracks = {}, sel_tracks = {}, splits = 0, deletes = 0,
        inserted_paths = {}, edit_cur = 0, msg = nil, undo = nil,
        forbidden = {}, ripple = {}, commands = {} }
  setup(W)
  reaper = make_reaper()
  local fn = assert(loadfile(SCRIPT))
  fn()
  return W
end

local function track(items)
  local t = { items = {} }
  for _, it in ipairs(items) do t.items[#t.items + 1] = new_item(it[1], it[2]) end
  return t
end

local pass, fail = 0, 0
local function check(name, cond, detail)
  if cond then pass = pass + 1; print(("  PASS  %s"):format(name))
  else fail = fail + 1; print(("  FAIL  %s  -- %s"):format(name, detail or "")) end
end

local function layout(tr)
  local s = {}
  for _, it in ipairs(tr.items) do
    s[#s + 1] = ("%s[%.2f..%.2f]"):format(it.tag == "tone" and "T" or "A", it.pos, it.pos + it.len)
  end
  return table.concat(s, " ")
end

print("\n1) Selection inside one long item -> split x2, middle deleted, tone inserted")
local w = run(function(W)
  local t = track({ {0, 30} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
end)
local t1 = w.tracks[1]
check("two splits", w.splits == 2, "splits=" .. w.splits)
check("one delete", w.deletes == 1, "deletes=" .. w.deletes)
check("tone inserted", w.last_inserted ~= nil)
check("tone spans selection", math.abs(w.last_inserted.pos - 10) < 1e-6
      and math.abs(w.last_inserted.len - 5) < 1e-6,
      ("pos=%.3f len=%.3f"):format(w.last_inserted.pos, w.last_inserted.len))
check("fades applied", w.last_inserted.fin_ > 0 and w.last_inserted.fout_ > 0)
check("no audio left inside range", (function()
  for _, it in ipairs(t1.items) do
    if it.tag == "audio" and it.pos >= 10 - 1e-6 and it.pos + it.len <= 15 + 1e-6 then return false end
  end
  return true
end)())
print("     layout: " .. layout(t1))

print("\n2) Item entirely inside selection -> deleted outright")
w = run(function(W)
  local t = track({ {11, 2} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
end)
check("no splits needed", w.splits == 0, "splits=" .. w.splits)
check("deleted", w.deletes == 1, "deletes=" .. w.deletes)
print("     layout: " .. layout(w.tracks[1]))

print("\n3) Items straddling each edge only")
w = run(function(W)
  local t = track({ {5, 7}, {13, 6} })   -- 5..12 and 13..19, selection 10..15
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
end)
local t3 = w.tracks[1]
check("split each straddler once", w.splits == 2, "splits=" .. w.splits)
check("both inner halves deleted", w.deletes == 2, "deletes=" .. w.deletes)
check("audio before survives", (function()
  for _, it in ipairs(t3.items) do
    if it.tag == "audio" and math.abs(it.pos - 5) < 1e-6 and math.abs(it.len - 5) < 1e-6 then return true end
  end
  return false
end)())
print("     layout: " .. layout(t3))

print("\n4) Item entirely outside selection -> untouched")
w = run(function(W)
  local t = track({ {20, 5} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
end)
check("no splits", w.splits == 0)
check("no deletes", w.deletes == 0)
check("still one audio item + tone", #w.tracks[1].items == 2, "n=" .. #w.tracks[1].items)

print("\n5) Only the selected track is touched")
w = run(function(W)
  local a, b = track({ {0, 30} }), track({ {0, 30} })
  W.tracks = { a, b }; W.sel_tracks = { a }; W.sel_start, W.sel_end = 10, 15
end)
check("other track untouched", #w.tracks[2].items == 1, "n=" .. #w.tracks[2].items)
check("selected track modified", #w.tracks[1].items > 1)

print("\n6) Guard rails")
w = run(function(W)
  local t = track({ {0, 30} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 10  -- empty selection
end)
check("aborts with message on empty selection", w.msg ~= nil and w.splits == 0, tostring(w.msg))

w = run(function(W)
  local t = track({ {0, 30} })
  W.tracks = { t }; W.sel_tracks = {}; W.sel_start, W.sel_end = 10, 15     -- no track selected
end)
check("aborts with message on no track", w.msg ~= nil and w.splits == 0, tostring(w.msg))

print("\n7) Never uses project-wide / UI-level calls that move other tracks")
w = run(function(W)
  local a, b = track({ {0, 30} }), track({ {0, 30} })
  W.tracks = { a, b }; W.sel_tracks = { a }; W.sel_start, W.sel_end = 10, 15
end)
check("no InsertMedia (ripples + can spawn tracks)", not w.forbidden["InsertMedia"])
check("does not move edit cursor", not w.forbidden["SetEditCurPos"])
check("does not change track selection", not w.forbidden["SetOnlyTrackSelected"]
      and not w.forbidden["SetTrackSelected"])
check("built item from PCM source", w.src_path ~= nil and w.last_inserted.take ~= nil)
check("tone landed on the SELECTED track", w.last_inserted.owner == w.tracks[1])

print("\n8) Ripple editing forced off, then restored")
w = run(function(W)
  local t = track({ {0, 30} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
  W.ripple[40311] = true            -- user had "ripple all tracks" on
end)
check("turned ripple off first", w.commands[1] == 40309, "cmds=" .. table.concat(w.commands, ","))
check("restored ripple-all after", w.commands[#w.commands] == 40311,
      "cmds=" .. table.concat(w.commands, ","))

w = run(function(W)
  local t = track({ {0, 30} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
  W.ripple[40310] = true            -- per-track ripple
end)
check("restored per-track ripple", w.commands[1] == 40309 and w.commands[#w.commands] == 40310,
      "cmds=" .. table.concat(w.commands, ","))

w = run(function(W)
  local t = track({ {0, 30} })
  W.tracks = { t }; W.sel_tracks = { t }; W.sel_start, W.sel_end = 10, 15
end)
check("ripple already off -> no mode changes at all", #w.commands == 0,
      "cmds=" .. table.concat(w.commands, ","))

print(("\n%d passed, %d failed\n"):format(pass, fail))
os.exit(fail == 0 and 0 or 1)
