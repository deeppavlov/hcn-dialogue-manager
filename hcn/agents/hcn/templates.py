import copy


class Template(object):

    def __init__(self, default="", dontcare=""):
        self.default = default
        self.dontcare = dontcare

    @classmethod
    def from_str(cls, s):
        return cls(*s.split('\t', 1))

    def update(self, default="", dontcare=""):
        self.default = self.default or default
        self.dontcare = seld.dontcare or dontcare

    def __contains__(self, t):
        return t.default and (t.default == self.default)\
                or t.dontcare and (t.dontcare == self.dontcare)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.default == other.default)\
                    and (self.dontcare == other.dontcare)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Override the default hash behavior (that returns the id)"""
        return hash(self.default + '\t' + self.dontcare)

    def __str__(self):
        return self.default + '\t' + self.dontcare

    def generate_text(self, slots):
        t = copy.copy(self.default)
        if any(s[1] == 'dontcare' for s in slots):
            t = copy.copy(self.dontcare)
        if t is None:
            print("Error while filling template {} with slots {}"\
                  .format(self.default or self.dontcare, slots))
            return ""
        if type(slots) == dict:
            slots = slots.items()
        for slot, value in slots:
            t = t.replace('#' + slot, value, 1)
        if t:
            t = t[0].upper() + t[1:]
        return t


class Templates(object):

    def __init__(self):
        self.act2templ = {}
        self.templ2act = {}
        self._actions = []
        self._templates = []

    def __contains__(self, key):
        """If key is an str, returns whether the key is in the actions.
        If key is a Template, returns if the key is templates.
        """
        if type(key) == str:
            return key in self.act2templ
        elif type(key) == Template:
            return key in self.templ2act

    def __getitem__(self, key):
        """If key is an str, returns corresponding template.
        If key is a Template, return corresponding action.
        If does not exist, return None.
        """
        if type(key) == str:
            return self.act2templ[key]
        elif type(key) == Template:
            return self.templ2act[key]

    def __len__(self):
        return len(self.act2templ)

    def __str__(self):
        return str(self.act2templ)

    def __setitem__(self, key, value):
        """If the key is not in  the dictionary, add it."""
        key = str(key)
        if key not in self.act2templ:
            self.act2templ[key] = value
            self.templ2act[value] = key
            self._actions = []
            self._templates = []

    @property
    def actions(self):
        if not self._actions:
            self._actions = sorted(self.act2templ.keys())
        return self._actions

    @property
    def templates(self):
        if not self._templates:
            self._templates = [self.act2templ[a] for a in self.actions]
        return self._templates

    def load(self, filename):
        for ln in open(filename, 'r'):
            act, template = ln.strip('\n').split('\t', 1)
            self.__setitem__(act, Template.from_str(template))
        return self

    def save(self, filename):
        with open(filename, 'w') as outfile:
            for act in sorted(self.actions):
                template = self.__getitem__(act)
                outfile.write('{}\t{}\n'.format(act, template))
