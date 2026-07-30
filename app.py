import os
import json
import logging
from datetime import datetime, timedelta
import streamlit as st
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configurazione Log e Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="LangGraph Smart Assistant", page_icon="📅")
st.title("📅 Assistente Appuntamenti Architettato in JSON")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("⚠️ Chiave API di OpenAI mancante su Render!")
    st.stop()

# Inizializzazione LLM rigido (temperature=0)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

JSON_FILE = "calendario.json"

# --- STRUTTURA PYDANTIC PER IL PARSING DI OPENAI ---
class UserIntent(BaseModel):
    intento: str = Field(description="Classificazione rigorosa dell'azione dell'utente: 'PRENOTAZIONE', 'CONFERMA', 'INFO', 'ALTRO'")
    orizzonte_temporale: str = Field(description="Il periodo espresso dall'utente: 'questa_settimana', 'prossima_settimana', 'giorno_specifico', 'generico'")
    giorno_indicato: Optional[str] = Field(None, description="Il giorno della settimana o data menzionata, convertito in minuscolo italiano (es. 'venerdì', 'martedì')")
    ora_indicata: Optional[str] = Field(None, description="L'orario preciso menzionato (es. '10:30')")

# Creiamo un parser LLM strutturato che restituisce SOLO la classe Pydantic/JSON
llm_strutturato = llm.with_structured_output(UserIntent)

# --- FUNZIONI LOCALI DI GESTIONE TEMPO E DATI ---
def get_current_context() -> str:
    giorni = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
    oggi = datetime.now()
    return f"Contesto Temporale: Oggi è {giorni[oggi.weekday()]} {oggi.strftime('%Y-%m-%d')}."

def leggi_calendario() -> list:
    if not os.path.exists(JSON_FILE):
        return []
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def scrivi_calendario(dati: list):
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(dati, f, indent=2)

def filtra_slot_python(orizzonte: str, giorno: Optional[str]) -> list:
    """Filtra matematicamente in Python le date del JSON evitando fraintendimenti del LLM"""
    oggi = datetime.now()
    slot_totali = leggi_calendario()
    slot_liberi = [s for s in slot_totali if s["disponibile"]]
    
    # Calcolo inizio e fine prossima settimana (Prossimo Lunedì)
    giorni_a_lunedi = (0 - oggi.weekday()) % 7
    if giorni_a_lunedi == 0: 
        giorni_a_lunedi = 7
    inizio_prossima = (oggi + timedelta(days=giorni_a_lunedi)).replace(hour=0, minute=0, second=0)
    fine_prossima = inizio_prossima + timedelta(days=7)
    
    risultato = []
    giorni_mappa = {0: "lunedì", 1: "martedì", 2: "mercoledì", 3: "giovedì", 4: "venerdì", 5: "sabato", 6: "domenica"}
    
    for s in slot_liberi:
        dt_slot = datetime.fromisoformat(s["data_ora"])
        giorno_sett_slot = giorni_mappa[dt_slot.weekday()]
        
        # Filtro temporale rigido basato sui timestamp Python
        if orizzonte == "prossima_settimana" and not (inizio_prossima <= dt_slot < fine_prossima):
            continue
        if orizzonte == "questa_settimana" and not (oggi <= dt_slot < inizio_prossima):
            continue
        if giorno and giorno.lower() != giorno_sett_slot:
            continue
            
        risultato.append(s)
    return risultato

# --- DEFINIZIONE DELLO STATO COMPOSITO ---
class State(TypedDict):
    user_input: str
    booking_status: str
    graph_action: str
    bot_response: str
    dati_estratti: Dict  # Salva il JSON strutturato estratto dall'utente nei vari passaggi

# --- DEFINIZIONE DEI NODI ---
def router_node(state: State) -> Dict:
    """NLU Puro: Trasforma il testo dell'utente in un JSON strutturato stabile"""
    contesto = get_current_context()
    prompt = f"""{contesto}
    Analizza la richiesta dell'utente e popola la struttura JSON dei dati estratti.
    Testo utente: "{state['user_input']}"
    """
    try:
        estrazione: UserIntent = llm_strutturato.invoke([HumanMessage(content=prompt)])
        dati_json = estrazione.model_dump()
    except Exception:
        logger.exception("Fallimento estrazione strutturata")
        dati_json = {"intento": "ALTRO", "orizzonte_temporale": "generico", "giorno_indicato": None, "ora_indicata": None}
    
    # Decidi il percorso in base all'intento estratto nel JSON
    mappa_azioni = {"CONFERMA": "vai_a_conferma", "PRENOTAZIONE": "vai_a_booking", "INFO": "vai_a_info"}
    azione = mappa_azioni.get(dati_json["intento"], "vai_a_fallback")
    
    # FIX: Corretto 'action' in 'azione'
    return {"graph_action": azione, "dati_estratti": dati_json}

def booking_node(state: State) -> Dict:
    """Prende i dati filtrati matematicamente da Python e genera la risposta d'aiuto"""
    dati = state["dati_estratti"]
    slot_filtrati = filtra_slot_python(dati["orizzonte_temporale"], dati["giorno_indicato"])
    
    if not slot_filtrati:
        return {"bot_response": "Mi dispiace, ma al momento non abbiamo slot liberi per il periodo richiesto. Desideri valutare altre date?"}
        
    # Costruiamo una stringa leggibile partendo dagli oggetti JSON filtrati da Python
    stringa_opzioni = ""
    for s in slot_filtrati:
        dt = datetime.fromisoformat(s["data_ora"])
        stringa_opzioni += f"- {dt.strftime('%A %d %B alle ore %H:%M')} (ID: {s['id']})\n"
        
    prompt = f"Genera un messaggio cordiale proponendo all'utente ESCLUSIVAMENTE queste opzioni disponibili:\n{stringa_opzioni}\nChiedi quale preferisce confermare."
    risposta = llm.invoke([HumanMessage(content=prompt)]).content
    return {"bot_response": risposta, "booking_status": "In corso"}

def conferma_node(state: State) -> Dict:
    """Esegue la cancellazione atomica modificando il JSON strutturato"""
    dati = state["dati_estratti"]
    # Troviamo lo slot filtrando per giorno ed eventualmente ora passati nel JSON dello stato
    slot_idonei = filtra_slot_python(dati["orizzonte_temporale"], dati["giorno_indicato"])
    
    if dati["ora_indicata"]:
        slot_idonei = [s for s in slot_idonei if dati["ora_indicata"] in s["data_ora"]]
        
    if not slot_idonei:
        return {"bot_response": "Non sono riuscito a trovare uno slot corrispondente libero nel nostro sistema. Puoi indicarmi un orario tra quelli proposti?"}
        
    # Se c'è ambiguità o troppi slot per quel giorno e l'utente non ha specificato l'ora
    if len(slot_idonei) > 1 and not dati["ora_indicata"]:
        stringa_orari = ", ".join([datetime.fromisoformat(s["data_ora"]).strftime("%H:%M") for s in slot_idonei])
        return {"bot_response": f"Per quel giorno ho più orari disponibili ({stringa_orari}). Quale preferisci di preciso?"}
        
    # FIX: Selezioniamo correttamente il primo elemento della lista
    slot_scelto = slot_idonei[0]
    tutti_i_dati = leggi_calendario()
    for s in tutti_i_dati:
        if s["id"] == slot_scelto["id"]:
            s["disponibile"] = False
            break
            
    scrivi_calendario(tutti_i_dati)
    dt_confermata = datetime.fromisoformat(slot_scelto["data_ora"]).strftime("%d %B alle %H:%M")
    
    return {
        "booking_status": "CONFERMATA",
        "bot_response": f"🎉 Ottimo! Ho registrato la tua scelta. Il tuo appuntamento è fissato per **{dt_confermata}**. Lo slot è stato bloccato."
    }

def info_node(state: State) -> Dict:
    slot_liberi = filtra_slot_python("tutti", None)
    prompt = f"Fornisci un riassunto sintetico basandoti solo su questi ID e date disponibili nel gestionale:\n{str(slot_liberi)}"
    risposta = llm.invoke([HumanMessage(content=prompt)]).content
    return {"bot_response": risposta}

def fallback_node(state: State) -> Dict:
    risposta = llm.invoke([SystemMessage(content="Rispondi brevemente e con cortesia."), HumanMessage(content=state['user_input'])]).content
    return {"bot_response": risposta}

# --- LOGICA BORDI E GRAFO ---
def route_decision(state: State) -> str:
    mappa = {"vai_a_conferma": "conferma", "vai_a_booking": "booking", "vai_a_info": "info"}
    return mappa.get(state.get("graph_action"), "fallback")

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

# --- CHAT STREAMLIT ---
if "messages" not in st.session_state: st.session_state.messages = []
if "booking_status" not in st.session_state: st.session_state.booking_status = "Nessuna"
if "storico_dati" not in st.session_state: st.session_state.storico_dati = {}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt_utente := st.chat_input("Scrivi qui..."):
    with st.chat_message("user"): st.markdown(prompt_utente)
    st.session_state.messages.append({"role": "user", "content": prompt_utente})
    
    stato_iniziale = {
        "user_input": prompt_utente,
        "booking_status": st.session_state.booking_status,
        "graph_action": "", "bot_response": "",
        "dati_estratti": st.session_state.storico_dati
    }
    
    with st.spinner("🤖 Elaborazione deterministica in corso..."):
        risultato = graph.invoke(stato_iniziale)
        
