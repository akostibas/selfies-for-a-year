-- Bulk-fetch as many useful properties as we can
tell application "Music"
	set lib to library playlist 1
	set pcs to played count of every track of lib
	set scs to skipped count of every track of lib
	set rats to rating of every track of lib
	set rks to rating kind of every track of lib
	set yrs to year of every track of lib
	set arts to artist of every track of lib
	set nms to name of every track of lib
	set dadd to date added of every track of lib
	set dpl to played date of every track of lib
	set durs to duration of every track of lib
	set gens to genre of every track of lib
end tell

set n to count of pcs
set out to ""
repeat with i from 1 to n
	set pc to item i of pcs
	set sc to item i of scs
	if pc > 0 or sc > 0 then
		set rt to item i of rats
		set rk to item i of rks as string
		set yr to item i of yrs
		set ar to item i of arts
		set nm to item i of nms
		set da to item i of dadd
		try
			set dp to (item i of dpl) as string
		on error
			set dp to ""
		end try
		set du to item i of durs
		set ge to item i of gens
		set out to out & pc & tab & sc & tab & rt & tab & rk & tab & yr & tab & (da as string) & tab & dp & tab & du & tab & ge & tab & ar & tab & nm & linefeed
	end if
end repeat
return out
