-- Same as before but include POSIX path; drop streaming-only tracks
tell application "Music"
	set lib to library playlist 1
	set pcs to played count of every track of lib
	set scs to skipped count of every track of lib
	set rats to rating of every track of lib
	set rks to rating kind of every track of lib
	set yrs to year of every track of lib
	set arts to artist of every track of lib
	set nms to name of every track of lib
	set dpl to played date of every track of lib
	set durs to duration of every track of lib
	set alls to every track of lib
end tell

set n to count of pcs
set out to ""
repeat with i from 1 to n
	set pc to item i of pcs
	set sc to item i of scs
	if pc > 0 or sc > 0 then
		try
			tell application "Music"
				set p to (get location of (item i of alls))
			end tell
			set ppath to POSIX path of p
		on error
			set ppath to "" -- streaming only
		end try
		if ppath is not "" then
			set rt to item i of rats
			set rk to item i of rks as string
			set yr to item i of yrs
			set ar to item i of arts
			set nm to item i of nms
			try
				set dp to (item i of dpl) as string
			on error
				set dp to ""
			end try
			set du to item i of durs
			set out to out & pc & tab & sc & tab & rt & tab & rk & tab & yr & tab & dp & tab & du & tab & ar & tab & nm & tab & ppath & linefeed
		end if
	end if
end repeat
return out
