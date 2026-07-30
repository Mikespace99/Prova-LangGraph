import os
import streamlit as st
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configurazione interfaccia grafica Streamlit
st.set_page_config(page_title="LangGraph OpenAI Chat", page_icon="🤖")
st.title("🤖 LangGraph + OpenAI Assistant")
st.write("Assistente intelligente distribuito su Render.")

# Recupero della chiave API dalle variabili d'ambiente (impostata su Render)
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ Chiave API di OpenAI mancante! Impostala nelle Environment Variables su Render.")
    st.stop()

# Inizializzazione del modello OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# --- 1. DEFINIZIONE DELLO STATO ---
class State(TypedDict):
    user_input: str
    booking_status: str
    graph_action: str
    bot_response: str

# --- 2. DEFINIZIONE DEI NODI ---
def router_node(state: State) -> Dict:
    """Usa OpenAI per capire l'intenzione dell'utente in modo versatile (NLU)"""
    prompt = f"""Analizza il messaggio dell'utente e rispondi ESCLUSIVAMENTE con una di queste tre parole:
    - 'PRENOTAZIONE': se l'utente vuole fissare, bloccare, spostare un appuntamento o riservare un posto.
    - 'INFO': se l'utente chiede la disponibilità, gli orari, i prezzi o se c'è posto.
    - 'ALTRO': se l'utente saluta, ringrazia, fa domande generiche o non inerenti.
    
    Messaggio utente: "{state['user_input']}"
    Risposta:"""
    
    risposta_llm = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    
    if "PRENOTAZIONE" in risposta_llm:
        return {"graph_action": "vai_a_booking"}
    elif "INFO" in risposta_llm:
        return {"graph_action": "vai_a_info"}
    else:
        return {"graph_action": "vai_a_fallback"}

def booking_node(state: State) -> Dict:
    prompt = f"Genera una risposta accogliente dicendo che stai avviando la procedura di prenotazione per la sua richiesta: '{state['user_input']}'"
    risposta = llm.invoke([HumanMessage(content=prompt)]).content
    return {"booking_status": "In corso", "bot_response": risposta}

def info_node(state: State) -> Dict:
    prompt = f"Rispondi all'utente dicendo che c'è ampia disponibilità per la prossima settimana e rispondi alla sua domanda: '{state['user_input']}'"
    risposta = llm.invoke([HumanMessage(content=prompt)]).content
    return {"bot_response": risposta}

def fallback_node(state: State) -> Dict:
    """Gestisce i saluti o le chiacchiere generali direttamente con l'LLM"""
    prompt = [
        SystemMessage(content="Sei un assistente virtuale cordiale per la gestione di appuntamenti. Rispondi in modo naturale."),
        HumanMessage(content=state['user_input'])
    ]
    risposta = llm.invoke(prompt).content
    return {"bot_response": risposta}

# --- 3. LOGICA DEI BORDI CONDIZIONALI ---
def route_decision(state: State) -> str:
    azione = state.get("graph_action")
    if azione == "vai_a_booking":
        return "booking"
    elif azione == "vai_a_info":
        return "info"
    else:
        return "fallback"

# --- 4. COSTRUZIONE DEL GRAFO ---
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("booking", booking_node)
builder.add_node("info", info_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "booking": "booking",
        "info": "info",
        "fallback": "fallback"
    }
)
builder.add_edge("booking", END)
builder.add_edge("info", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# --- 5. INTERFACCIA CHAT STREAMLIT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra i messaggi storici della chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
if prompt_utente := st.chat_input("Come posso aiutarti?"):
    with st.chat_message("user"):
        st.markdown(prompt_utente)
    st.session_state.messages.append({"role": "user", "content": prompt_utente})

    # Inizializza lo stato per LangGraph
    stato_corrente = {
        "user_input": prompt_utente,
        "booking_status": "Nessuna",
        "graph_action": "",
        "bot_response": ""
    }

    # Esegui il grafo
    con ricarica_animazione = st.spinner("🤖 Sto pensando..."):
        risultato_grafo = graph.invoke(stato_corrente)
        risposta_finale = risultato_grafo["bot_response"]

    # Mostra la risposta del Bot
    with st.chat_message("assistant"):
        st.markdown(risposta_finale)
    st.session_state.messages.append({"role": "assistant", "content": risposta_finale})
