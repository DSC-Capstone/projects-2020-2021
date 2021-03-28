class bare_sentence:
    def __init__(self):
        self.senetence_id=None
        self.noun = []
        self.verb = []
        self.ner = []
class bare_document:
    def __init__(self):
        self.doc_id = None
        self.path = None
        self.sentences = {}
class bare_collection:
    clo_id = None
    doc = {}
    dir_path = None
class bare_parssed_struct:
    id=None
    process=None
    docu=None
    source=''
    sourceClass=''
    language=''
    dictionary=None
    indexStruct=None











