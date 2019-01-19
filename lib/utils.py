################################################################################

class Opener:
    """
    This class allows unify working with sys.stdin or a file given by its filename
    which is a string. This class will take care of opening and closing a stream
    if the argument is a filename.
    USAGE:
     > with utils.Opener(file_or_stdin) as f:

    Python: simple things made too complex
    """

    def __init__(self, fname):
        self.fname = fname
        self.i_have_opened_it = False
        self.channel = None

    def __enter__(self):
        if isinstance(self.fname, str):
            self.channel = open(self.fname)
            self.i_have_opened_it = True
        else:
            self.channel = self.fname
        return self.channel

    def __exit__(self, type, value, traceback):
        if self.i_have_opened_it:
            self.channel.close()

################################################################################

