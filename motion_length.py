"""Short app defaults, independent of historical experiment lengths."""
SHORT = 'Short · 3.2 s'
STANDARD = 'Standard · 3.9 s'
LONG = 'Long · 7.7 s'
COUNTS = {SHORT: 81, STANDARD: 97, LONG: 193}

def choices(middle=None):
    return [SHORT, STANDARD] + ([LONG] if middle else [])

def frames(choice=STANDARD, middle=None):
    if choice not in choices(middle):
        raise ValueError('Choose Short or Standard; Long requires a Middle frame.')
    return COUNTS[choice]
