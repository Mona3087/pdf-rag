This Chainlit app was created following instructions from [this repository!]()
    [ User Query ]
        ||
        \/
    [ ChatOpenAI ] -------> [ OpenAIEmbeddings ]
        ||                          ||
        \/                          \/
    [ Qdrant ] <------ [ PyMuPDFLoader ]
        ||                ||      ||
        \/                \/      \/
    [ PDF Document ] <--- [ Text Extraction & Processing ]
        ||                               ||
        \/                               \/
    [ Text Chunking ] ----> [ Vector Storage ]
        ||                               ||
        \/                               \/
    [ Vector Retrieval ] <-- [ Text Embeddings ]
        ||                               ||
        \/                               \/
    [ Answer Generation ]
        ||
        \/
    [ User Answer ]
