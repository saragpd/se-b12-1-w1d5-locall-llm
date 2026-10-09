import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# Thisharvi AI Configuration
# -------------------------------

MODEL_PATH = "./models/Qwen2.5-0.5B-Instruct"

st.set_page_config(
    page_title="Thisharvi AI",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# Custom UI Styling
# -------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .main-title {
        text-align: center;
        color: #153e75;
        font-size: 38px;
        font-weight: 800;
    }

    .subtitle {
        text-align: center;
        color: #52657a;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .trust-note {
        text-align: center;
        color: #167647;
        font-size: 13px;
        padding: 10px;
        background-color: #e7f7ee;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------

st.markdown(
    '<div class="main-title">🤖 Thisharvi AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your AI Assistant, Running Locally</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="trust-note">'
    '🔒 Local model inference • No external AI API required'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Brand tagline: Thisharvi — No. 1 Safest AI in India"
)

# -------------------------------
# Load Local Model
# -------------------------------

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="cpu",
        local_files_only=True
    )

    model.eval()

    return tokenizer, model


try:
    with st.spinner("Loading your local AI model..."):
        tokenizer, model = load_model()

    st.success("Thisharvi AI is ready!")

except Exception as e:
    st.error("Unable to load the local model.")
    st.code(str(e))
    st.info(
        "Check that MODEL_PATH points to the downloaded "
        "Qwen2.5-0.5B-Instruct folder."
    )
    st.stop()

# -------------------------------
# Chat History
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("**Model:** Qwen2.5-0.5B-Instruct")
    st.write("**Runtime:** Local CPU")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Thisharvi AI | Local LLM Project")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Chat Input and Generation
# -------------------------------

if prompt := st.chat_input("Ask Thisharvi AI anything..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Keep recent conversation context
    messages = [
        {
            "role": "system",
            "content": (
                "You are Thisharvi AI, a helpful and friendly "
                "AI assistant. Answer clearly and honestly. "
                "If you are uncertain, say so."
            )
        }
    ]

    messages.extend(st.session_state.messages[-10:])

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thisharvi AI is thinking..."):

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    text,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id
                    )

                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )

                st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:
        st.error(f"Response generation failed: {e}")