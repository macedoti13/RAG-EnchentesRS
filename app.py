import gradio as gr
from rag import RAG

list_urls = [
    "https://g1.globo.com/rs/rio-grande-do-sul/noticia/2024/05/15/levantamento-enchente-porto-alegre-bairros-e-moradores.ghtml",
    "https://www.cnnbrasil.com.br/nacional/enchente-em-porto-alegre-veja-imagens-da-cidade-apos-queda-no-nivel-da-agua/",
    "https://www.bbc.com/portuguese/articles/cw00d51k5rlo",
    "https://gauchazh.clicrbs.com.br/ultimas-noticias/tag/alagamentos/",
    "https://www.poder360.com.br/infraestrutura/especialistas-listam-medidas-para-evitar-enchentes-em-porto-alegre/",
    "https://www.camarapoa.rs.gov.br/noticias/cece-discute-sobre-obras-nas-escolas-atingidas-pelas-enchentes",
    "https://www.metropoles.com/brasil/sobe-para-179-o-numero-de-mortos-em-razao-das-enchentes-no-rs",
    "https://gauchazh.clicrbs.com.br/economia/conteudo-de-marca/2024/06/pequenos-negocios-podem-receber-ate-15-mil-reais-para-custos-com-alagamentos-clxdskw6i01420144abiw0pvv.html",
    "https://agenciabrasil.ebc.com.br/geral/noticia/2024-05/inundacao-em-porto-alegre-foi-falta-de-manutencao-dizem-especialistas",
    "https://www.observatoriodasmetropoles.net.br/nucleo-porto-alegre-analisa-os-impactos-das-enchentes-na-populacao-pobre-e-negra-do-rio-grande-do-sul/",
    "https://gauchazh.clicrbs.com.br/porto-alegre/noticia/2024/05/numero-de-abrigos-para-atingidos-pela-enchente-cai-em-porto-alegre-clwdpjnrp00vg0148idb5123y.html",
    "https://gauchazh.clicrbs.com.br/porto-alegre/noticia/2024/05/diversos-bairros-de-porto-alegre-registram-inundacao-dmae-fala-em-chuva-alem-do-que-os-modelos-previam-clwjbt6rh00b1014xqkk45ji9.html#:~:text=Entre%20os%20registros%20de%20aumento,São%20Geraldo%2C%20Restinga%20e%20Floresta."
]

rag = None

def initialize_rag(completion_model, embedding_model):
    global rag
    rag = RAG(completion_model=completion_model, embedding_model=embedding_model)
    rag.add_documents(list_urls)
    return "Models initialized with completion_model: {} and embedding_model: {}".format(completion_model, embedding_model)

def add_documents(links):
    if rag is None:
        return "Please initialize the models first."
    urls = [url.strip() for url in links.split(",")]
    rag.add_documents(urls)
    return f"Added {len(urls)} document(s) to the RAG."

def get_response(question):
    if rag is None:
        return "Please initialize the models first."
    response = rag.query(question)
    return response

def get_context(question):
    if rag is None:
        return "Please initialize the models first."
    documents = rag.retrieved_contexts.get(question, [])
    context_content = "\n\n".join([doc.page_content for doc in documents])
    return context_content if context_content else "No context found."

with gr.Blocks() as ui:
    gr.Markdown("# RAG sobre as enchentes no Rio Grande do Sul")
    
    with gr.Row():
        with gr.Column(scale=1):
            completion_model_dropdown = gr.Dropdown(
                choices=["gpt-3.5-turbo", "gpt-4o"], 
                label="Select Completion Model"
            )
            embedding_model_dropdown = gr.Dropdown(
                choices=["text-embedding-3-small", "text-embedding-3-large"], 
                label="Select Embedding Model"
            )
            initialize_button = gr.Button("Initialize Models")
            initialize_output = gr.Textbox(label="Initialization Status", lines=2, interactive=False)
            
            initialize_button.click(
                fn=initialize_rag, 
                inputs=[completion_model_dropdown, embedding_model_dropdown], 
                outputs=initialize_output
            )
        
        with gr.Column(scale=1):
            add_links_input = gr.Textbox(label="Add document links (comma separated)", lines=2)
            add_links_button = gr.Button("Add Documents")
            add_links_output = gr.Textbox(label="Add Links Status", lines=2, interactive=False)
            
            add_links_button.click(
                fn=add_documents, 
                inputs=add_links_input, 
                outputs=add_links_output
            )
    
    question_input = gr.Textbox(label="Type your question here", lines=2)
    response_button = gr.Button("Get Response")
    response_output = gr.Textbox(label="Response", lines=10, interactive=False)
    
    context_button = gr.Button("Check Context")
    context_output = gr.Textbox(label="Context", lines=10, interactive=False)
    
    response_button.click(fn=get_response, inputs=question_input, outputs=response_output)
    context_button.click(fn=get_context, inputs=question_input, outputs=context_output)

    with gr.Row():
        with gr.Column():
            question_input
            response_button
            response_output
        with gr.Column():
            context_button
            context_output

ui.launch()
