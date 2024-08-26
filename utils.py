from __future__ import annotations

import io
import os
import re
import subprocess
import textwrap
import time
import uuid
import wave

import emoji
import gradio as gr
import langid
import nltk
import numpy as np
import noisereduce as nr
from huggingface_hub import HfApi

# Download the 'punkt' tokenizer for the NLTK library
nltk.download("punkt")

# will use api to restart space on a unrecoverable error
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = os.environ.get("REPO_ID")
api = HfApi(token=HF_TOKEN)

latent_map = {}

def get_latents(chatbot_voice, xtts_model, voice_cleanup=False):
    global latent_map
    if chatbot_voice not in latent_map:    
        speaker_wav = f"examples/{chatbot_voice}.wav"
        if (voice_cleanup):
            try:
                cleanup_filter="lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02" 
                resample_filter="-ac 1 -ar 22050"
                out_filename = speaker_wav + str(uuid.uuid4()) + ".wav"  #ffmpeg to know output format
                #we will use newer ffmpeg as that has afftn denoise filter
                shell_command = f"ffmpeg -y -i {speaker_wav} -af {cleanup_filter} {resample_filter} {out_filename}".split(" ")
                command_result = subprocess.run([item for item in shell_command], capture_output=False,text=True, check=True)
                speaker_wav=out_filename
                print("Filtered microphone input")
            except subprocess.CalledProcessError:
                # There was an error - command exited with non-zero code
                print("Error: failed filtering, use original microphone input")
        else:
                speaker_wav=speaker_wav
        # gets condition latents from the model
        # returns tuple (gpt_cond_latent, speaker_embedding)
        latent_map[chatbot_voice] = xtts_model.get_conditioning_latents(audio_path=speaker_wav)
    return latent_map[chatbot_voice]

  
def detect_language(prompt, xtts_supported_languages=None):
    if xtts_supported_languages is None:
        xtts_supported_languages = ["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","ja"] 

    # Fast language autodetection
    if len(prompt)>15:
        language_predicted=langid.classify(prompt)[0].strip() # strip need as there is space at end!
        if language_predicted == "zh": 
            #we use zh-cn on xtts
            language_predicted = "zh-cn"
            
        if language_predicted not in xtts_supported_languages:
            print(f"Detected a language not supported by xtts :{language_predicted}, switching to english for now")
            gr.Warning(f"Language detected '{language_predicted}' can not be spoken properly 'yet' ")
            language= "en"
        else:
            language = language_predicted
        print(f"Language: Predicted sentence language:{language_predicted} , using language for xtts:{language}")
    else:
        # Hard to detect language fast in short sentence, use english default
        language = "en"
        print(f"Language: Prompt is short or autodetect language disabled using english for xtts")

    return language
    
def get_voice_streaming(prompt, language, chatbot_voice, xtts_model, suffix="0"):
    gpt_cond_latent, speaker_embedding = get_latents(chatbot_voice, xtts_model) 
    try:
        t0 = time.time()
        chunks = xtts_model.inference_stream(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=7.0,
            temperature=0.85,
        )

        first_chunk = True
        for i, chunk in enumerate(chunks):
            if first_chunk:
                first_chunk_time = time.time() - t0
                metrics_text = f"Latency to first audio chunk: {round(first_chunk_time*1000)} milliseconds\n"
                first_chunk = False
            #print(f"Received chunk {i} of audio length {chunk.shape[-1]}")

            # In case output is required to be multiple voice files
            # out_file = f'{char}_{i}.wav'
            # write(out_file, 24000, chunk.detach().cpu().numpy().squeeze())
            # audio = AudioSegment.from_file(out_file)
            # audio.export(out_file, format='wav')
            # return out_file
            # directly return chunk as bytes for streaming
            chunk = chunk.detach().cpu().numpy().squeeze()
            chunk = (chunk * 32767).astype(np.int16)
            yield chunk.tobytes()

    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(
                f"Exit due to: Unrecoverable exception caused by prompt:{prompt}",
                flush=True,
            )
            gr.Warning("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")

            # HF Space specific.. This error is unrecoverable need to restart space
            api.restart_space(REPO_ID=REPO_ID)
        else:
            print("RuntimeError: non device-side assert error:", str(e))
            # Does not require warning happens on empty chunk and at end
            ###gr.Warning("Unhandled Exception encounter, please retry in a minute")
            return None
        return None
    except:
        return None

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

def format_prompt(message, history):
    system_message = f"""
    You are an empathetic, insightful, and supportive coach who helps people deal with challenges and celebrate achievements.
    You help people feel better by asking questions to reflect on and evoke feelings of positivity, gratitude, joy, and love.
    You show radical candor and tough love.
    Respond in a casual and friendly tone.
    Sprinkle in filler words, contractions, idioms, and other casual speech that we use in conversation.
    Emulate the user’s speaking style and be concise in your response.
    """
    prompt = (
        "<s>[INST]" + system_message + "[/INST]"
    )
    for user_prompt, bot_response in history:
        if user_prompt is not None:
            prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    
    if message=="":
        message="Hello"
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate_llm_output(
        prompt,    
        history,
        llm,
        temperature=0.8,
        max_tokens=256,
        top_p=0.95,
        stop_words=["<s>","[/INST]", "</s>"]
    ):
        temperature = float(temperature)
        if temperature < 1e-2:
            temperature = 1e-2
        top_p = float(top_p)

        generate_kwargs = dict(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop_words
        )
        formatted_prompt = format_prompt(prompt, history)
        try:
            print("LLM Input:", formatted_prompt)
            # Local GGUF
            stream = llm(
                formatted_prompt,
                **generate_kwargs,
                stream=True,
            )
            output = ""
            for response in stream:
                character= response["choices"][0]["text"]

                if character in stop_words:
                    # end of context
                    return 
                    
                if emoji.is_emoji(character):
                    # Bad emoji not a meaning messes chat from next lines
                    return
                
                output += response["choices"][0]["text"]
                yield output

        except Exception as e:
            print("Unhandled Exception: ", str(e))
            gr.Warning("Unfortunately Mistral is unable to process")
            output = "I do not know what happened but I could not understand you ."
        return output
    
def get_sentence(history, llm):
    history = [["", None]] if history is None else history 
    history[-1][1] = ""        
    sentence_list = []
    sentence_hash_list = []

    text_to_generate = ""
    stored_sentence = None
    stored_sentence_hash = None
    
    for character in generate_llm_output(history[-1][0], history[:-1], llm):
        history[-1][1] = character.replace("<|assistant|>","")
        # It is coming word by word
        text_to_generate = nltk.sent_tokenize(history[-1][1].replace("\n", " ").replace("<|assistant|>"," ").replace("<|ass>","").replace("[/ASST]","").replace("[/ASSI]","").replace("[/ASS]","").replace("","").strip())
        if len(text_to_generate) > 1:
            
            dif = len(text_to_generate) - len(sentence_list)

            if dif == 1 and len(sentence_list) != 0:
                continue

            if dif == 2 and len(sentence_list) != 0 and stored_sentence is not None:
                continue

            # All this complexity due to trying append first short sentence to next one for proper language auto-detect
            if stored_sentence is not None and stored_sentence_hash is None and dif>1:
                #means we consumed stored sentence and should look at next sentence to generate
                sentence = text_to_generate[len(sentence_list)+1]
            elif stored_sentence is not None and len(text_to_generate)>2 and stored_sentence_hash is not None:
                print("Appending stored")
                sentence = stored_sentence + text_to_generate[len(sentence_list)+1]
                stored_sentence_hash = None
            else:
                sentence = text_to_generate[len(sentence_list)]
                
            # too short sentence just append to next one if there is any
            # this is for proper language detection 
            if len(sentence)<=15 and stored_sentence_hash is None and stored_sentence is None:
                if sentence[-1] in [".","!","?"]:
                    if stored_sentence_hash != hash(sentence):
                        stored_sentence = sentence
                        stored_sentence_hash = hash(sentence) 
                        print("Storing:",stored_sentence)
                        continue
            
            
            sentence_hash = hash(sentence)
            if stored_sentence_hash is not None and sentence_hash == stored_sentence_hash:
                continue
            
            if sentence_hash not in sentence_hash_list:
                sentence_hash_list.append(sentence_hash)
                sentence_list.append(sentence)
                print("New Sentence: ", sentence)
                yield (sentence, history)

    # return that final sentence token
    try:
        last_sentence = nltk.sent_tokenize(history[-1][1].replace("\n", " ").replace("<|ass>","").replace("[/ASST]","").replace("[/ASSI]","").replace("[/ASS]","").replace("","").strip())[-1]
        sentence_hash = hash(last_sentence)
        if sentence_hash not in sentence_hash_list:
            if stored_sentence is not None and stored_sentence_hash is not None:
                last_sentence = stored_sentence + last_sentence
                stored_sentence = stored_sentence_hash = None
                print("Last Sentence with stored:",last_sentence)
        
            sentence_hash_list.append(sentence_hash)
            sentence_list.append(last_sentence)
            print("Last Sentence: ", last_sentence)
    
            yield (last_sentence, history)
    except:
        print("ERROR on last sentence history is :", history)
            
# will generate speech audio file per sentence
def generate_speech_for_sentence(history, chatbot_voice, sentence, xtts_model, xtts_supported_languages=None, filter_output=True, return_as_byte=False):
    language = "autodetect"

    wav_bytestream = b""
    
    if len(sentence)==0:
        print("EMPTY SENTENCE")
        return 
    
    # Sometimes prompt </s> coming on output remove it
    # Some post process for speech only
    sentence = sentence.replace("</s>", "")
    # remove code from speech
    sentence = re.sub("```.*```", "", sentence, flags=re.DOTALL)
    sentence = re.sub("`.*`", "", sentence, flags=re.DOTALL)
    
    sentence = re.sub("\(.*\)", "", sentence, flags=re.DOTALL)
    
    sentence = sentence.replace("```", "")
    sentence = sentence.replace("...", " ")
    sentence = sentence.replace("(", " ")
    sentence = sentence.replace(")", " ")
    sentence = sentence.replace("<|assistant|>","")

    if len(sentence)==0:
        print("EMPTY SENTENCE after processing")
        return 
        
    # A fast fix for last chacter, may produce weird sounds if it is with text
    #if (sentence[-1] in ["!", "?", ".", ","]) or (sentence[-2] in ["!", "?", ".", ","]):
    #    # just add a space
    #    sentence = sentence[:-1] + " " + sentence[-1]
        
    # regex does the job well
    sentence= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?|\!)",r"\1 \2\2",sentence)
    
    print("Sentence for speech:", sentence)

    
    try:
        SENTENCE_SPLIT_LENGTH=350
        if len(sentence)<SENTENCE_SPLIT_LENGTH:
            # no problem continue on
            sentence_list = [sentence]
        else:
            # Until now nltk likely split sentences properly but we need additional 
            # check for longer sentence and split at last possible position
            # Do whatever necessary, first break at hypens then spaces and then even split very long words
            sentence_list=textwrap.wrap(sentence,SENTENCE_SPLIT_LENGTH)
            print("SPLITTED LONG SENTENCE:",sentence_list)
        
        for sentence in sentence_list:
            
            if any(c.isalnum() for c in sentence):
                if language=="autodetect":
                    #on first call autodetect, nexts sentence calls will use same language
                    language = detect_language(sentence, xtts_supported_languages) 
            
                #exists at least 1 alphanumeric (utf-8) 
                audio_stream = get_voice_streaming(
                        sentence, language, chatbot_voice, xtts_model
                    )
            else:
                # likely got a ' or " or some other text without alphanumeric in it
                audio_stream = None 
                
            # XTTS is actually using streaming response but we are playing audio by sentence
            # If you want direct XTTS voice streaming (send each chunk to voice ) you may set DIRECT_STREAM=1 environment variable
            if audio_stream is not None:
                frame_length = 0
                for chunk in audio_stream:
                    try:
                        wav_bytestream += chunk
                        frame_length += len(chunk)
                    except:
                        # hack to continue on playing. sometimes last chunk is empty , will be fixed on next TTS
                        continue

            # Filter output for better voice
            if filter_output:
                data_s16 = np.frombuffer(wav_bytestream, dtype=np.int16, count=len(wav_bytestream)//2, offset=0)
                float_data = data_s16 * 0.5**15
                reduced_noise = nr.reduce_noise(y=float_data, sr=24000,prop_decrease =0.8,n_fft=1024)
                wav_bytestream = (reduced_noise * 32767).astype(np.int16)
                wav_bytestream = wav_bytestream.tobytes()
                    
            if audio_stream is not None:
                if not return_as_byte:
                    audio_unique_filename = "/tmp/"+ str(uuid.uuid4())+".wav"
                    with wave.open(audio_unique_filename, "w") as f:
                        f.setnchannels(1)
                        # 2 bytes per sample.
                        f.setsampwidth(2)
                        f.setframerate(24000)
                        f.writeframes(wav_bytestream)
                           
                    return (history , gr.Audio.update(value=audio_unique_filename, autoplay=True))
                else:
                    return (history , gr.Audio.update(value=wav_bytestream, autoplay=True))
    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            print(
                f"Exit due to: Unrecoverable exception caused by prompt:{sentence}",
                flush=True,
            )
            gr.Warning("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")

            # HF Space specific.. This error is unrecoverable need to restart space
            api.restart_space(REPO_ID=REPO_ID)
        else:
            print("RuntimeError: non device-side assert error:", str(e))
            raise e

    print("All speech ended")
    return 
