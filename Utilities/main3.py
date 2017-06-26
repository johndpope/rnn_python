from Utilities import music_utilities
from Utilities import constants

do = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
do[0] += 1
do[18] += 1
do[6] += 1

do1 = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
do1[0] += 1
do1[1] += 1
do1[6] += 1

do2 = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
do2[0] += 2
do2[6] += 1

si = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
si[0] += 1
si[17] += 1
si[5] += 1

re = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
re[0] += 1
re[20] += 1
re[8] += 1

la = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
la[0] += 1
la[15] += 1
la[3] += 1

end = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
end[0] += 1
end[1] += 2

end1 = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
end1[0] += 2
end1[1] += 1

pause = [0] * ((constants.MAX_PITCH - constants.MIN_PITCH) + 2)
pause[0] += 3

music_utilities.generate_from_sequence([do, do1, do2, end1, si, si, si, end, do, end, do, end, pause, pause, re, re, re, end, re, re, re, end, si, end, si, end, pause, pause, la, la, la, end])