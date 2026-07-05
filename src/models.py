from dataclasses import dataclass, fields

@dataclass
class Document: #One loaded source file
    doc_id:str
    source_path:str
    source_name:str
    fmt:str # file type "md" | "txt" | "html"
    raw_text:str
    clean_text:str
    metadata:dict
    
@dataclass
class Chunk:
    chunk_id:str
    doc_id:str
    source_name:str
    chunk_index: int # position within the doc
    text: str
    section_heading: str | None # nearest markdown heading above this chunk
    chunking_strategy: str
    char_count: int
    token_count: int 
    start_char: int
    end_char: int
    
    def to_metadata(self) -> dict:
        
        result = {}
        for f in fields(self):
            name = f.name # iterate through attribute
            value = getattr(self,name) #grab value of attribute
            if name != "text": # only store when name doesn't equal text
                result[name]= value
                
        return result
            
            

            
            
