import os
import streamlit as st
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configurazione interfaccia grafica Streamlit
st.set_page_config(page_title="LangGraph TXT Booking", page_icon="🤖")
st.title("🤖 LangGraph + Auto-Delete TXT Calendar")
st.write("Le prenotazioni confermate vengono rimosse automaticamente dal file calendario.txt.")

# Recupero della chiave API dalle variabili d'ambiente
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ Chiave API di OpenAI mancante! Impostala su Render.")
    st.stop()

# Inizializzazione del modello OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# --- FUNZIONI DI GESTIONE DEL FILE TXT ---
TXT_FILE = "calendario.txt"

def get_free_slots_from_txt() -> str:
    """Legge il file calendario.txt e restituisce tutti gli slot ancora disponibili"""
    if not os.path.exists(TXT_FILE):
        return "Nessuno slot inserito nel sistema."
        
    elenco_liberi = ""
    with open(TXT_FILE, "r", encoding="utf-8") as file:
        for riga in file:
            riga = riga.strip()
            if riga:
                elenco_liberi += f"- {riga}\n"
                
    if not elenco_liberi:
        return "Tutti gli slot sono esauriti."
        
    return elenco_liberi

def remove_slot_from_txt(slot_da_cancellare: str) -> bool:
    """Riscrive il file rimuovendo la riga dello slot prenotato"""
    if not os.path.exists(TXT_FILE) or not slot_da_cancellare:
        return False
        
    righe_da_salvare = []
    trovato = False
    
    with open(TXT_FILE, "r", encoding="utf-8") as file:
        for riga in file:
            riga_pulita = riga.strip()
            # Se la riga del file è contenuta o corrisponde allo slot identificato dall'LLM, la saltiamo (cancella)
            if riga_pulita and riga_pulita.lower() in slot_da_cancellare.lower():
                trovato = True
            elif riga_pulita:
                righe_da_salvare.append(riga_pulita)
                
    # Riscrive il file solo con le righe rimaste
    with open(TXT_FILE, "w", encoding="utf-8") as file:
        for riga in righe_da_salvare:
            file.write(riga + "\n")
            
    return trovato


# --- 1. DEFINIZIONE DELLO STATO ---
class State(TypedDict):
    user_input: str
    booking_status: str  # "Nessuna", "In corso", "CONFERMATA"
    graph_action: str
    bot_response: str
    slot_selezionato: str # Memorizza lo slot specifico da cancellare nel file


# --- 2. DEFINIZIONE DEI NODI ---
def router_node(state: State) -> Dict:
    """Analizza il testo per capire se l'utente sta CONFERMANDO uno slot preciso o se vuole solo info"""
    slot_reali = get_free_slots_from_txt()
    
    prompt = f"""Analizza il messaggio dell'utente e decidi l'azione corretta da prendere.
    
    SLOT ANCORA DISPONIBILI NEL SISTEMA:
    {slot_reali}
    
    Messaggio utente: "{state['user_input']}"
    
    Rispondi ESCLUSIVAMENTE con una di queste tre parole:
    - 'CONFERMA': se l'utente accetta, sceglie o conferma chiaramente uno degli slot sopra indicati (es: "Prendo lunedì alle 11:30", "Ok per martedì 15:30", "confermo per venerdì").
    - 'PRENOTAZIONE': se l'utente esprime solo il desiderio generale di prenotare o chiede quali sono i posti liberi.
    - 'ALTRO': se saluta, ringrazia o fa domande non inerenti.
    
    Risposta:"""
    
    risposta_llm = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    
    if "CONFERMA" in risposta_llm:
        return {"graph_action": "vai_a_conferma"}
    elif "PRENOTAZIONE" in risposta_llm:
        return {"graph_action": "vai_a_booking"}
    else:
        return {"graph_action": "vai_a_fallback"}

def booking_node(state: State) -> Dict:
    """Mostra i posti liberi leggendo dal file txt"""
    slot_reali_liberi = get_free_slots_from_txt()
    
    prompt = [
        SystemMessage(content=(
            "Sei l'assistente per le prenotazioni.\n"
            "Proponi ESCLUSIVAMENTE gli slot presenti in questa lista forniti in tempo reale dal nostro registro. "
            "Se l'utente ha chiesto un giorno specifico, isola solo quelli di quel giorno.\n\n"
            f"LISTA SLOT DISPONIBILI:\n{slot_reali_liberi}\n"
            "Chiedi esplicitamente all'utente di confermare quale orario preferisce accettare."
        )),
        HumanMessage(content=state['user_input'])
    ]
    risposta = llm.invoke(prompt).content
    return {"booking_status": "In corso", "bot_response": risposta, "slot_selezionato": ""}

def conferma_node(state: State) -> Dict:
    """Nodo critico: estrae la riga esatta scelta dall'utente, la cancella dal file e conferma"""
    slot_reali_liberi = get_free_slots_from_txt()
    
    # Chiediamo a OpenAI di estrarre la stringa ESATTA presente nel file txt che l'utente ha scelto
    prompt_estrazione = f"""Trova quale riga della lista corrisponde alla scelta dell'utente.
    LISTA DISPONIBILE NEL FILE:
    {slot_reali_liberi}
    
    Scelta dell'utente: "{state['user_input']}"
    
    Rispondi ESCLUSIVAMENTE con la riga esatta presa dalla lista (es: "Venerdì - 10:30"). Se non trovi una corrispondenza esatta, rispondi 'ERRORE'.
    Risposta:"""
    
    stringa_slot = llm.invoke([HumanMessage(content=prompt_estrazione)]).content.strip()
    
    if "ERRORE" not in stringa_slot:
        # Eseguiamo la cancellazione fisica dal file txt
        successo = remove_slot_from_txt(stringa_slot)
        if successo:
            risposta_conferma = f"🎉 Perfetto! Ho registrato la tua prenotazione per **{stringa_slot}**. Lo slot è stato ufficialmente bloccato."
            return {"booking_status": "CONFERMATA", "bot_response": risposta_conferma, "slot_selezionato": stringa_slot}
            
    risposta_errore = "Non sono riuscito a trovare o bloccare l'orario richiesto. Potrebbe essere stato appena occupato. Ti dispiace scegliere un altro slot?"
    return {"booking_status": "In corso", "bot_response": risposta_errore, "slot_selezionato": ""}

def fallback_node(state: State) -> Dict:
    """Gestisce chiacchiere o saluti generali"""
    prompt = [
        SystemMessage(content="Sei un assistente virtuale cordiale. Rispondi in modo naturale e conciso."),
        HumanMessage(content=state['user_input'])
    ]
    risposta = llm.invoke(prompt).content
    return {"bot_response": risposta}


# --- 3. LOGICA DEI BORDI CONDIZIONALI ---
def route_decision(state: State) -> str:
    azione = state.get("graph_action")
    if azione == "vai_a_conferma":
        return "conferma"
    elif azione == "vai_a_booking":
        return "booking"
    else:
        return "fallback"


# --- 4. COSTRUZIONE DEL GRAFO ---
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("booking", booking_node)
builder.add_node("conferma", conferma_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "booking": "booking",
        "conferma": "conferma",
        "fallback": "fallback"
    }
)
builder.add_edge("booking", END)
builder.add_edge("conferma", END)
builder.add_edge("fallback", END)

graph = builder.compile()


# --- 5. INTERFACCIA CHAT STREAMLIT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mantieni lo stato della prenotazione nella sessione dell'app
if "booking_status" not in st.session_state:
    st.session_state.booking_status = "Nessuna"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_utente := st.chat_input("Scrivi qui..."):
    with st.chat_message("user"):
        st.markdown(prompt_utente)
    st.session_state.messages.append({"role": "user", "content": prompt_utente})

    # Passiamo lo stato corrente a LangGraph
    stato_iniziale = {
        "user_input": prompt_utente,
        "booking_status": st.session_state.booking_status,
        "graph_action": "",
        "bot_response": "",
        "slot_selezionato": ""
    }

    with st.spinner("🤖 Controllo ed elaborazione del file..."):
        risultato_grafo = graph.invoke(stato_iniziale)
        risposta_finale = risultato_grafo["bot_response"]
        st.session_state.booking_status = risultato_grafo["booking_status"]

    with st.chat_message("assistant"):
        st.markdown(risposta_finale)
    st.session_state.messages.append({"role": "assistant", "content": risposta_finale})
