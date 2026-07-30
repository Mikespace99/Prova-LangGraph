import logging
from typing import Dict

logger = logging.getLogger(__name__)


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
- PRENOTAZIONE: l'utente chiede informazioni generali sugli slot o esprime intenzione di prenotare senza indicare una scelta precisa.
- ALTRO: saluti, ringraziamenti o richieste non pertinenti.

Rispondi con una sola parola tra: CONFERMA, PRENOTAZIONE, ALTRO."""

    try:
        risposta = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    except Exception:
        logger.exception("Errore nella chiamata LLM in router_node")
        return {"graph_action": "vai_a_fallback"}

    if "CONFERMA" in risposta:
        return {"graph_action": "vai_a_conferma"}
    if "PRENOTAZIONE" in risposta:
        return {"graph_action": "vai_a_booking"}
    return {"graph_action": "vai_a_fallback"}


def conferma_node(state: State) -> Dict:
    """Estrae lo slot scelto dall'utente e lo rimuove dal registro disponibilità.

    A differenza della versione precedente, la riga restituita dal modello
    viene validata contro la lista reale degli slot prima di essere usata:
    un output del LLM non corrispondente esattamente a una riga esistente
    viene trattato come mancata identificazione, non come conferma valida.
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

    slot_validi = [riga.strip() for riga in slot_disponibili.splitlines() if riga.strip()]

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
