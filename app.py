import os
import logging
from datetime import datetime
import streamlit as st
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configurazione logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Interfaccia Streamlit
st.set_page_config(page_title="LangGraph Smart Calendar", page_icon="📅")
st.title("📅 Assistente Appuntamenti con Datetime")
st.write("Flusso d'azione LangGraph con calcolo del tempo reale in Python.")

# Recupero chiave API
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ Chiave API di OpenAI mancante! Impostala su Render.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

TXT_FILE = "calendario.txt"

# --- CALCOLO DELLA DATA CORRENTE IN PYTHON ---
def get_current_time_context() -> str:
    """Restituisce il giorno della settimana e la data odierna per ancorare l'LLM"""
    giorni_settimana = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]
    ora_attuale = datetime.now()
    nome_giorno = giorni_settimana[ora_attuale.weekday()]
    return f"Oggi è {nome_giorno}, data: {ora_attuale.strftime('%Y-%m-%d')} (Formato: AAAA-MM-GG)."

def get_free_slots_from_txt() -> str:
    """Legge il file calendario.txt e restituisce tutti gli slot liberi strutturati"""
    if not os.path.exists(TXT_FILE):
        return "Nessuno slot inserito nel sistema."
        
    elenco_liberi = ""
    with open(TXT_FILE, "r", encoding="utf-8") as file:
        for riga in file:
            riga = riga.strip()
            if riga:
                elenco_liberi += f"{riga}\n"
                
    if not elenco_liberi.strip():
        return "Tutti gli slot sono esauriti."
        
    return elenco_liberi

def remove_slot_from_txt(slot_da_cancellare: str) -> bool:
    """Riscrive il file rimuovendo la riga esatta dello slot prenotato"""
    if not os.path.exists(TXT_FILE) or not slot_da_cancellare:
        return False
        
    righe_da_salvare = []
    trovato = False
    
    with open(TXT_FILE, "r", encoding="utf-8") as file:
        for riga in file:
            riga_pulita = riga.strip()
            if riga_pulita and riga_pulita.lower() in slot_da_cancellare.lower():
                trovato = True
            elif riga_pulita:
                righe_da_salvare.append(riga_pulita)
                
    with open(TXT_FILE, "w", encoding="utf-8") as file:
        for riga in righe_da_salvare:
            file.write(riga + "\n")
            
    return trovato

# --- 1. DEFINIZIONE DELLO STATO ---
class State(TypedDict):
    user_input: str
    booking_status: str     
    graph_action: str       
    bot_response: str
    slot_selezionato: str   

# --- 2. DEFINIZIONE DEI NODI ---
def router_node(state: State) -> Dict:
    """Classifica l'intento calcolando le date relative espresse dall'utente"""
    slot_disponibili = get_free_slots_from_txt()
    contesto_temporale = get_current_time_context()
    
    prompt = f"""{contesto_temporale}
    
    SLOT ANCORA DISPONIBILI NEL REGISTRO (Formato AAAA-MM-GG OORA:MM):
    {slot_disponibili}
    
    MESSAGGIO UTENTE:
    "{state['user_input']}"
    
    Regole di classificazione:
    - CONFERMA: l'utente sceglie uno slot specifico. Se usa termini come "domani", "prossimo venerdì", "venerdì va bene", usa il contesto temporale per capire a quale data numerica corrisponde. Se corrisponde a uno slot in lista, rispondi CONFERMA.
    - PRENOTAZIONE: l'utente vuole fare una prenotazione generica o chiede la lista dei posti liberi.
    - INFO: l'utente fa domande generali.
    - ALTRO: saluti o chiacchiere.
    
    Rispondi ESCLUSIVAMENTE con una parola tra: CONFERMA, PRENOTAZIONE, INFO, ALTRO."""
    
    try:
        risposta = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    except Exception:
        logger.exception("Errore in router_node")
        return {"graph_action": "vai_a_fallback"}
        
    if "CONFERMA" in risposta:
        return {"graph_action": "vai_a_conferma"}
    if "PRENOTAZIONE" in risposta:
        return {"graph_action": "vai_a_booking"}
    if "INFO" in risposta:
        return {"graph_action": "vai_a_info"}
    return {"graph_action": "vai_a_fallback"}

def booking_node(state: State) -> Dict:
    slot_reali_liberi = get_free_slots_from_txt()
    contesto_temporale = get_current_time_context()
    
    prompt = [
        SystemMessage(content=(
            f"{contesto_temporale}\n"
            "Sei l'assistente per le prenotazioni. Mostra gli slot traducendo i codici AAAA-MM-GG in giorni leggibili "
            "(es. 'Venerdì 31 Luglio alle 09:00') per l'utente, ma mantieni internamente il riferimento rigido.\n\n"
            f"LISTA SLOT DISPONIBILI:\n{slot_reali_liberi}\n"
            "Chiedi all'utente di confermare quale preferisce."
        )),
        HumanMessage(content=state['user_input'])
    ]
    try:
        risposta = llm.invoke(prompt).content
    except Exception:
        logger.exception("Errore in booking_node")
        risposta = "Errore nel caricamento degli slot."
    return {"booking_status": "In corso", "bot_response": risposta, "slot_selezionato": ""}

def info_node(state: State) -> Dict:
    slot_reali_liberi = get_free_slots_from_txt()
    prompt = [
        SystemMessage(content=f"Fornisci un riassunto delle disponibilità basandoti solo su questo elenco:\n{slot_reali_liberi}"),
        HumanMessage(content=state['user_input'])
    ]
    try:
        risposta = llm.invoke(prompt).content
    except Exception:
        risposta = "Errore di lettura."
    return {"bot_response": risposta}

def conferma_node(state: State) -> Dict:
    """Identifica la data calcolata, la valida e la elimina dal file txt"""
    slot_disponibili = get_free_slots_from_txt()
    contesto_temporale = get_current_time_context()
    
    prompt = f"""{contesto_temporale}
    LISTA DISPONIBILE:
    {slot_disponibili}
    
    SCELTA UTENTE:
    "{state['user_input']}"
    
    Converti la scelta dell'utente (es. 'domani', 'venerdì') nella riga esatta così come appare nella LISTA DISPONIBILE (es. '2026-07-31 09:00').
    Rispondi esclusivamente con la stringa esatta trovata in lista, altrimenti scrivi 'ERRORE'."""
    
    try:
        slot_estratto = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        logger.exception("Errore in conferma_node")
        return {"booking_status": "In corso", "bot_response": "Errore temporaneo.", "slot_selezionato": ""}
        
    slot_validi = [riga.strip() for riga in slot_disponibili.splitlines() if riga.strip()]
    
    match_trovato = None
    for sv in slot_validi:
        if slot_estratto.lower() in sv.lower() or sv.lower() in slot_estratto.lower():
            match_trovato = sv
            break
            
    if not match_trovato or "ERRORE" in slot_estratto:
        return {
            "booking_status": "In corso",
            "bot_response": "Non ho trovato una corrispondenza esatta per quel giorno. Puoi specificare meglio data e ora?",
            "slot_selezionato": ""
        }
        
    successo = remove_slot_from_txt(match_trovato)
    if not successo:
        return {
            "booking_status": "In corso",
            "bot_response": "Slot non disponibile.",
            "slot_selezionato": ""
        }
        
    return {
        "booking_status": "CONFERMATA",
        "bot_response": f"🎉 Prenotazione confermata per lo slot **{match_trovato}**. L'orario è stato rimosso dal sistema!",
        "slot_selezionato": match_trovato
    }

def fallback_node(state: State) -> Dict:
    prompt = [SystemMessage(content="Rispondi in modo naturale e conciso."), HumanMessage(content=state['user_input'])]
    try:
        risposta = llm.invoke(prompt).content
    except Exception:
        risposta = "Sono qui."
    return {"bot_response": risposta}

# --- 3. LOGICA DEI BORDI ---
def route_decision(state: State) -> str:
    azione = state.get("graph_action")
    if azione == "vai_a_conferma": return "conferma"
    elif azione == "vai_a_booking": return "booking"
    elif azione == "vai_a_info": return "info"
    else: return "fallback"

# --- 4. COSTRUZIONE DEL GRAFO ---
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("booking", booking_node)
builder.add_node("info", info_node)
builder.add_node("conferma", conferma_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_decision, {"booking": "booking", "info": "info", "conferma": "conferma", "fallback": "fallback"})
builder.add_edge("booking", END)
builder.add_edge("info", END)
builder.add_edge("conferma", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# --- 5. INTERFACCIA CHAT STREAMLIT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "booking_status" not in st.session_state: st.session_state.booking_status = "Nessuna"

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt_utente := st.chat_input("Scrivi qui..."):
    with st.chat_message("user"): st.markdown(prompt_utente)
    st.session_state.messages.append({"role": "user", "content": prompt_utente})

    stato_iniziale = {
        "user_input": prompt_utente,
        "booking_status": st.session_state.booking_status,
        "graph_action": "", "bot_response": "", "slot_selezionato": ""
    }

    with st.spinner("🤖 Elaborazione..."):
        risultato_grafo = graph.invoke(stato_iniziale)
        risposta_finale = risultato_grafo["bot_response"]
        st.session_state.booking_status = risultato_grafo["booking_status"]

    with st.chat_message("assistant"): st.markdown(risposta_finale)
