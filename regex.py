import re

pattern = pattern = r"Schedule"

text = "Schedule 'hi I am tars' to +123456789 at '2023-12-06 01:10'."

if re.match(pattern, text) is not None:
    print(True)
else:

    print(False)


