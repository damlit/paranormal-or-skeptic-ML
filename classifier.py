#!/usr/bin/python3

import sys
import re

pRegex = r'ufo|paranormal|youtube'
sRegex = r'science|skeptic|placebo'

for line in sys.stdin:
    if re.search(sRegex, line.lower()):
        print(' S')
    elif re.search(pRegex, line.lower()):
        print(' P')
    else:
        print(' S')
