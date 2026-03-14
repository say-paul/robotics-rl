#!/bin/bash
# Patch noVNC rfb.js to skip unknown encodings instead of disconnecting
FILE=/usr/share/novnc/core/rfb.js

# Replace the _fail call for unsupported encoding with a warning + skip
python3 -c "
import re
with open('$FILE') as f:
    code = f.read()

old = '''            this._fail(\"Unsupported encoding (encoding: \" +
                       this._FBU.encoding + \")\");
            return false;'''

new = '''            Log.Warn(\"Skipping unsupported encoding: \" + this._FBU.encoding);
            return true;'''

if old in code:
    code = code.replace(old, new)
    with open('$FILE', 'w') as f:
        f.write(code)
    print('PATCHED rfb.js successfully')
else:
    print('WARNING: patch target not found')
"
