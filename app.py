

###### Set Up Environment ######

import os
# Set CUDA environment variable and install llama-cpp-python
# llama-cpp-python is a python binding for llama.cpp library which enables LLM inference in pure C/C++
os.environ["CUDACXX"] = "/usr/local/cuda/bin/nvcc"
os.system('python -m unidic download')
os.system('CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.11 --verbose')


# Third-party library imports
from faster_whisper import WhisperModel
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

# Local imports
from utils import get_sentence, generate_speech_for_sentence, wave_header_chunk

# Load Whisper ASR model
print("Loading Whisper ASR")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Load Mistral LLM
print("Loading Mistral LLM")
hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", local_dir=".", filename="mistral-7b-instruct-v0.1.Q5_K_M.gguf")
mistral_model_path="./mistral-7b-instruct-v0.1.Q5_K_M.gguf"
mistral_llm = Llama(model_path=mistral_model_path,n_gpu_layers=35,max_new_tokens=256, context_window=4096, n_ctx=4096,n_batch=128,verbose=False)


# Load XTTS Model
print("Loading XTTS model")
os.environ["COQUI_TOS_AGREED"] = "1"
tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(tts_model_name)
tts_model_path = os.path.join(get_user_data_dir("tts"), tts_model_name.replace("/", "--"))
config = XttsConfig()
config.load_json(os.path.join(tts_model_path, "config.json"))
xtts_model = Xtts.init_from_config(config)
xtts_model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(tts_model_path, "model.pth"),
    vocab_path=os.path.join(tts_model_path, "vocab.json"),
    eval=True,
    use_deepspeed=True,
)
xtts_model.cuda()

###### Set up Gradio Interface ######

with gr.Blocks(title="Voice chat with LLM") as demo:
    DESCRIPTION = """# Voice chat with LLM"""
    gr.Markdown(DESCRIPTION)

    # Define chatbot component
    chatbot = gr.Chatbot(
        value=[(None, "Hi friend, I'm Amy, an AI coach. How can I help you today?")],  # Initial greeting from the chatbot
        elem_id="chatbot",
        avatar_images=("examples/hf-logo.png", "examples/ai-chat-logo.png"),
        bubble_full_width=False,
    )

    # Define chatbot voice component
    VOICES = ["female", "male"]
    with gr.Row():
        chatbot_voice = gr.Dropdown(
            label="Voice of the Chatbot",
            info="How should Chatbot talk like",
            choices=VOICES,
            max_choices=1,
            value=VOICES[0],
        )

    # Define text and audio record input components
    with gr.Row():
        txt_box = gr.Textbox(
            scale=3,
            show_label=False,
            placeholder="Enter text and press enter, or speak to your microphone",
            container=False,
            interactive=True,
        )
        audio_record = gr.Audio(source="microphone", type="filepath", scale=4)

    # Define generated audio playback component 
    with gr.Row():
        sentence = gr.Textbox(visible=False)
        audio_playback = gr.Audio(
            value=None,
            label="Generated audio response",
            streaming=True,
            autoplay=True,
            interactive=False,
            show_label=True,
        )

    # Will be triggered on text submit (will send to generate_speech)
    def add_text(chatbot_history, text):
        chatbot_history = [] if chatbot_history is None else chatbot_history
        chatbot_history = chatbot_history + [(text, None)]
        return chatbot_history, gr.update(value="", interactive=False)
    
    # Will be triggered on voice submit (will transribe and send to generate_speech)
    def add_audio(chatbot_history, audio):
        chatbot_history = [] if chatbot_history is None else chatbot_history
        # get result from whisper and strip it to delete begin and end space
        response, _ = whisper_model.transcribe(audio)
        text = list(response)[0].text.strip()
        print("Transcribed text:", text)
        chatbot_history = chatbot_history + [(text, None)]
        return chatbot_history, gr.update(value="", interactive=False)

    def generate_speech(chatbot_history, chatbot_voice, initial_greeting=False):
        # Start by yielding an initial empty audio to set up autoplay
        yield ("", chatbot_history, wave_header_chunk())

        # Helper function to handle the speech generation and yielding process
        def handle_speech_generation(sentence, chatbot_history, chatbot_voice):
            if sentence != "":
                print("Processing sentence")
                generated_speech = generate_speech_for_sentence(chatbot_history, chatbot_voice, sentence, xtts_model, xtts_supported_languages=config.languages, return_as_byte=True)
                if generated_speech is not None:
                    _, audio_dict = generated_speech
                    yield (sentence, chatbot_history, audio_dict["value"])

        if initial_greeting:
            # Process only the initial greeting if specified
            for _, sentence in chatbot_history:
                yield from handle_speech_generation(sentence, chatbot_history, chatbot_voice)
        else:
            # Continuously get and process sentences from a generator function
            for sentence, chatbot_history in get_sentence(chatbot_history, mistral_llm):
                print("Inserting sentence to queue")
                yield from handle_speech_generation(sentence, chatbot_history, chatbot_voice)

    txt_msg = txt_box.submit(fn=add_text, inputs=[chatbot, txt_box], outputs=[chatbot, txt_box], queue=False
                             ).then(fn=generate_speech,  inputs=[chatbot,chatbot_voice], outputs=[sentence, chatbot, audio_playback])

    txt_msg.then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=[txt_box], queue=False)

    audio_msg = audio_record.stop_recording(fn=add_audio, inputs=[chatbot, audio_record], outputs=[chatbot, txt_box], queue=False
                                            ).then(fn=generate_speech,  inputs=[chatbot,chatbot_voice], outputs=[sentence, chatbot, audio_playback])

    audio_msg.then(fn=lambda: (gr.update(interactive=True),gr.update(interactive=True,value=None)), inputs=None, outputs=[txt_box, audio_record], queue=False)

    FOOTNOTE = """
            This Space demonstrates how to speak to an llm chatbot, based solely on open accessible models.
            It relies on the following models :
            - Speech to Text Model: [Faster-Whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) an ASR model, to transcribe recorded audio to text.
            - Large Language Model: [Mistral-7b-instruct-v0.1-quantized](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) a LLM to generate the chatbot responses. 
            - Text to Speech Model: [XTTS-v2](https://huggingface.co/spaces/coqui/xtts) a TTS model, to generate the voice of the chatbot.

            Note:
            - Responses generated by chat model should not be assumed correct or taken serious, as this is a demonstration example only
            - iOS (Iphone/Ipad) devices may not experience voice due to autoplay being disabled on these devices by Vendor"""
    gr.Markdown(FOOTNOTE)
    demo.load(fn=generate_speech, inputs=[chatbot,chatbot_voice, gr.State(value=True)], outputs=[sentence, chatbot, audio_playback])
demo.queue().launch(debug=True,share=True)