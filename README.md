Tested on Python 3.6, OSX 10.12.

Pull submodules

    git submodule update --init

Run

    python bin/test.py

Build:

    cd matting
    make

Run tests:

    cd matting 
    py.test test


Visualization, launch a visdom server:

    python -m 'visdom.server'

### Caveats



### TODO
Once confident with basic tests:
- increase network capacity in `matting/modules.py::MattingCNNself.net`, put widht = 64, depth up to 10, grow_width=True
- if running out of memory, decrease cg_steps in the same MattingCNN class
