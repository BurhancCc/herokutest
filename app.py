import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000"  # Zorg dat je FastAPI op deze poort draait

st.set_page_config(page_title="Kenteken ↔️ Locatie Mapping", layout="wide")
st.title("🚛 Afvalstort Kentekenregistratie")

st.header("➕ Voeg een voertuig toe")

with st.form("voeg_toe_formulier"):
    kenteken = st.text_input("Kenteken").upper()
    locatiecode = st.text_input("Locatiecode (bijv. 101, 202)")
    omschrijving = st.text_input("Omschrijving (bijv. Bouwafval, Groenafval)")
    toevoegen = st.form_submit_button("Toevoegen")

    if toevoegen:
        if kenteken and locatiecode:
            payload = {
                "kenteken": kenteken,
                "locatiecode": locatiecode,
                "omschrijving": omschrijving
            }
            r = requests.post(f"{API_URL}/mappings", json=payload)
            if r.status_code == 200:
                st.success(f"✅ Toegevoegd: {kenteken} ↔ {locatiecode} ({omschrijving})")
            else:
                st.error("❌ Toevoegen mislukt.")
        else:
            st.warning("⚠️ Vul minimaal kenteken en locatiecode in.")

st.header("📋 Huidige registratie")
response = requests.get(f"{API_URL}/mappings")
if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)
    if not df.empty:
        df["Verwijderen"] = df["kenteken"].apply(
            lambda k: f"Verwijder {k}"
        )
        st.dataframe(df[["kenteken", "locatiecode", "omschrijving"]])
        
        with st.expander("🗑️ Verwijderen"):
            kentekens = df["kenteken"].tolist()
            selected = st.selectbox("Selecteer kenteken om te verwijderen", kentekens)
            if st.button("Verwijder geselecteerd kenteken"):
                del_response = requests.delete(f"{API_URL}/mappings/{selected}")
                if del_response.status_code == 200:
                    st.success(f"✅ {selected} verwijderd")
                    st.rerun()
                else:
                    st.error("❌ Verwijderen mislukt")

        if st.button("❌ Alles wissen"):
            if st.confirm("Weet je zeker dat je alles wilt verwijderen?"):
                clear_response = requests.delete(f"{API_URL}/mappings")
                if clear_response.status_code == 200:
                    st.success("✅ Alles verwijderd")
                    st.rerun()
    else:
        st.info("📭 Geen kentekens geregistreerd.")
else:
    st.error("❌ Kan mappings niet ophalen van de server.")

#===================================================================================
# Invoering kenteken uit afbeelding
#===================================================================================

st.header("📷 Controleer kenteken bij locatie")

with st.form("controle_formulier"):
    uploaded_image = st.file_uploader("Upload een foto van de vrachtwagen", type=["jpg", "jpeg", "png"])
    ingevoerde_locatiecode = st.text_input("Waar is de vrachtwagen gefotografeerd? (locatiecode)")
    verzenden = st.form_submit_button("Verwerk afbeelding")

    if verzenden:
        if uploaded_image and ingevoerde_locatiecode:
            # Voor nu slaan we gewoon op in sessie/geheugen
            st.session_state["laatste_upload"] = {
                "locatiecode": ingevoerde_locatiecode,
                "bestand": uploaded_image.name
            }
            st.success(f"✅ Bestand '{uploaded_image.name}' geregistreerd voor locatiecode {ingevoerde_locatiecode}")
            
            # Laat afbeelding zien (optioneel)
            st.image(uploaded_image, caption="Geüploade afbeelding", use_column_width=True)
        else:
            st.warning("⚠️ Voeg een afbeelding én locatiecode toe.")
