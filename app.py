import logging
import os
from typing import Dict, TypedDict

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="LangGraph TXT Chat", page_icon="🤖")
st.title("🤖 LangGraph + TXT Calendar Assistant")
st.write("Assistente connesso al file di testo calendario.txt.")

# --- CHIAVE API ---
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("⚠️ Chiave API di OpenAI mancante! Impostala su Render.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

# --- LETTURA / SCRITTURA DEL FILE calendario.txt ---
TXT_FILE = "calendario.txt"


def get_free_slots_from_txt() -> str:
    """Legge calendario.txt ed estrae solo le righe con stato LIBERO."""
    if not os.path.exists(TXT_FILE):
        return "Errore: Il file 'calendario.txt' non è stato trovato sul server."

    elenco_liberi = ""
    with open(TXT_FILE, "r", encoding="utf-8") as file:
        for riga in file:
            riga = riga.strip()
            if riga and "LIBERO" in riga.upper():
                pulita = riga.replace(" - LIBERO", "").replace(" - libero", "")
                elenco_liberi += f"- {pulita}\n"

    if not elenco_liberi:
        return "Nessuno slot disponibile al momento nel file."

    return elenco_liberi


def remove_slot_from_txt(slot: str) -> bool:
    """Marca come OCCUPATO lo slot corrispondente nel file calendario.txt.

    'slot' arriva nel formato pulito (es. 'Lunedì - 09:00'), lo stesso
    prodotto da get_free_slots_from_txt. Cerca la riga originale che
    contiene quel testo e la sostituisce, mantenendo LIBERO/OCCUPATO
    come stato coerente nel file.

    Ritorna True se la riga è stata trovata e aggiornata, False altrimenti
    (es. slot non esistente o già occupato).
    """
    if not os.path.exists(TXT_FILE):
        logger.error("calendario.txt non trovato durante remove_slot_from_txt")
        return False

    with open(TXT_FILE, "r", encoding="utf-8") as file:
        righe = file.readlines()

    aggiornato = False
    nuove_righe = []
    for riga in righe:
        riga_pulita = riga.strip()
        if riga_pulita.startswith(slot) and "LIBERO" in riga_pulita.upper():
            nuove_righe.append(f"{slot} - OCCUPATO\n")
            aggiornato = True
        else:
            nuove_righe.append(riga)

    if not aggiornato:
        return False

    with open(TXT_FILE, "w", encoding="utf-8") as file:
        file.writelines(nuove_righe)

    return True


# --- 1. STATO DEL GRAFO ---
class State(TypedDict):
    user_input: str
    booking_status: str
    graph_action: str
    bot_response: str
    slot_selezionato: str


# --- 2. NODI ---
def router_node(state: State) -> Dict:
    """Classifica l'intento dell'utente in base al messaggio e agli slot disponibili."""
    slot_disponibili = get_free_slots_from_txt()

    prompt = f"""Analizza il messaggio dell'utente e classifica l'azione richiesta.

SLOT DISPONIBILI:
{slot_disponibili}

MESSAGGIO UTENTE:
"{state['user_input']}"

Regole:
- CONFERMA: l'utente accetta o sceglie uno slot specifico, anche indicando solo il giorno se corrisponde a un unico slot disponibile.
- PRENOTAZIONE: l'utente esprime intenzione di prenotare senza indicare una scelta precisa e vuole essere guidato tra le opzioni.
- INFO: l'utente chiede la disponibilità complessiva o gli orari, senza intenzione immediata di prenotare.
- ALTRO: saluti, ringraziamenti o richieste non pertinenti.

Rispondi con una sola parola tra: CONFERMA, PRENOTAZIONE, INFO, ALTRO."""

    try:
        risposta = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    except Exception:
        logger.exception("Errore nella chiamata LLM in router_node")
        return {"graph_action": "vai_a_fallback"}

    if "CONFERMA" in risposta:
        return {"graph_action": "vai_a_conferma"}
    if "PRENOTAZIONE" in risposta:
        return {"graph_action": "vai_a_booking"}
    if "INFO" in risposta:
        return {"graph_action": "vai_a_info"}
    return {"graph_action": "vai_a_fallback"}


def booking_node(state: State) -> Dict:
    """Legge i dati veri dal file di testo e li passa all'LLM per proporre gli slot."""
    slot_reali_liberi = get_free_slots_from_txt()

    prompt = [
        SystemMessage(content=(
            "Sei l'assistente virtuale del sistema di prenotazione.\n"
            "Hai appena consultato il registro delle disponibilità in tempo reale.\n"
            "Proponi ESCLUSIVAMENTE gli slot realmente liberi indicati qui sotto. "
            "Non inventare opzioni non presenti in lista.\n\n"
            f"SLOT LIBERI DISPONIBILI:\n{slot_reali_liberi}\n"
            "REGOLE DI RISPOSTA:\n"
            "- Usa un tono professionale e neutro.\n"
            "- Mostra chiaramente l'elenco degli orari disponibili.\n"
            "- Se l'utente ha richiesto un giorno preciso, mostra solo gli orari di quel giorno.\n"
            "- Chiedi una conferma finale o una scelta specifica."
        )),
        HumanMessage(content=f"Messaggio attuale dell'utente: '{state['user_input']}'"),
    ]

    try:
        risposta = llm.invoke(prompt).content
    except Exception:
        logger.exception("Errore nella chiamata LLM in booking_node")
        risposta = "Si è verificato un problema temporaneo nel recuperare gli slot disponibili. Riprova tra poco."

    return {"booking_status": "In corso", "bot_response": risposta}


def conferma_node(state: State) -> Dict:
    """Estrae lo slot scelto dall'utente e lo marca come occupato nel registro.

    La riga restituita dal modello viene validata contro la lista reale
    degli slot prima di essere usata: un output del LLM non corrispondente
    esattamente a una riga esistente viene trattato come mancata
    identificazione, non come conferma valida.
    """
    slot_disponibili = get_free_slots_from_txt()

    prompt = f"""Individua a quale riga della lista corrisponde la scelta dell'utente.

LISTA DISPONIBILE:
{slot_disponibili}

SCELTA UTENTE:
"{state['user_input']}"

Se l'utente indica solo il giorno e la lista contiene un solo slot per quel giorno, seleziona quella riga.
Rispondi esclusivamente con la riga esatta copiata dalla lista, oppure con 'ERRORE' se non trovi corrispondenza."""

    try:
        slot_estratto = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        logger.exception("Errore nella chiamata LLM in conferma_node")
        return {
            "booking_status": "In corso",
            "bot_response": "Si è verificato un problema temporaneo. Puoi ripetere l'orario che preferisci?",
            "slot_selezionato": "",
        }

    slot_validi = [riga.strip().lstrip("- ") for riga in slot_disponibili.splitlines() if riga.strip()]

    if slot_estratto not in slot_validi:
        return {
            "booking_status": "In corso",
            "bot_response": "Non ho trovato una corrispondenza esatta con gli slot disponibili. Puoi indicare l'orario così come appare nell'elenco?",
            "slot_selezionato": "",
        }

    successo = remove_slot_from_txt(slot_estratto)

    if not successo:
        return {
            "booking_status": "In corso",
            "bot_response": "Quello slot risulta già occupato. Vuoi scegliere un altro orario tra quelli disponibili?",
            "slot_selezionato": "",
        }

    return {
        "booking_status": "CONFERMATA",
        "bot_response": f"Prenotazione confermata per {slot_estratto}.",
        "slot_selezionato": slot_estratto,
    }


def info_node(state: State) -> Dict:
    """Fornisce una panoramica generale basandosi sul file .txt."""
    slot_reali_liberi = get_free_slots_from_txt()
    prompt = [
        SystemMessage(content=(
            "Sei l'assistente informativo del calendario.\n"
            f"Basandoti su questi dati reali estratti dal registro:\n{slot_reali_liberi}\n"
            "Riassumi brevemente quali giorni hanno disponibilità e invita l'utente a scegliere una data per prenotare."
        )),
        HumanMessage(content=f"Domanda dell'utente: '{state['user_input']}'"),
    ]

    try:
        risposta = llm.invoke(prompt).content
    except Exception:
        logger.exception("Errore nella chiamata LLM in info_node")
        risposta = "Si è verificato un problema temporaneo nel recuperare le disponibilità. Riprova tra poco."

    return {"bot_response": risposta}


def fallback_node(state: State) -> Dict:
    """Gestisce chiacchiere o saluti generali."""
    prompt = [
        SystemMessage(content="Sei un assistente virtuale professionale. Rispondi in modo naturale, cordiale e conciso."),
        HumanMessage(content=state["user_input"]),
    ]

    try:
        risposta = llm.invoke(prompt).content
    except Exception:
        logger.exception("Errore nella chiamata LLM in fallback_node")
        risposta = "Si è verificato un problema temporaneo. Puoi ripetere la richiesta?"

    return {"bot_response": risposta}


# --- 3. LOGICA DI ROUTING CONDIZIONALE ---
def route_decision(state: State) -> str:
    azione = state.get("graph_action")
    if azione == "vai_a_conferma":
        return "conferma"
    if azione == "vai_a_booking":
        return "booking"
    if azione == "vai_a_info":
        return "info"
    return "fallback"


# --- 4. COSTRUZIONE DEL GRAFO ---
builder = StateGraph(State)
builder.add_node("router", router_node)
builder.add_node("booking", booking_node)
builder.add_node("conferma", conferma_node)
builder.add_node("info", info_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "conferma": "conferma",
        "booking": "booking",
        "info": "info",
        "fallback": "fallback",
    },
)
builder.add_edge("booking", END)
builder.add_edge("conferma", END)
builder.add_edge("info", END)
builder.add_edge("fallback", END)

graph = builder.compile()


# --- 5. INTERFACCIA CHAT STREAMLIT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_utente := st.chat_input("Come posso aiutarti?"):
    with st.chat_message("user"):
        st.markdown(prompt_utente)
    st.session_state.messages.append({"role": "user", "content": prompt_utente})

    stato_iniziale = {
        "user_input": prompt_utente,
        "booking_status": "Nessuna",
        "graph_action": "",
        "bot_response": "",
        "slot_selezionato": "",
    }

    with st.spinner("🤖 Elaborazione ed esecuzione grafo..."):
        risultato_grafo = graph.invoke(stato_iniziale)
        risposta_finale = risultato_grafo["bot_response"]

    with st.chat_message("assistant"):
        st.markdown(risposta_finale)
    st.session_state.messages.append({"role": "assistant", "content": risposta_finale})
