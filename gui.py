import os
import torch
import gradio as gr
import requests
import tqdm
from pathlib import Path
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

# Initialize directories
os.makedirs('outputs', exist_ok=True)
os.makedirs('processed', exist_ok=True)
os.makedirs('checkpoints/base_speakers', exist_ok=True)
os.makedirs('checkpoints/converter', exist_ok=True)

# Available languages and styles
LANGUAGES = {
    "English": "EN",
    "Chinese": "ZH",
    "Spanish": "ES",
    "French": "FR",
    "Japanese": "JP",
    "Korean": "KR",
}

STYLES = {
    "Default": "default",
    "Friendly": "friendly",
    "Cheerful": "cheerful", 
    "Excited": "excited", 
    "Sad": "sad", 
    "Angry": "angry", 
    "Terrified": "terrified", 
    "Shouting": "shouting", 
    "Whispering": "whispering"
}

# Initialize device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model URLs (Update these with actual URLs)
MODEL_CONFIG = {
    "converter": {
        "config": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/config.json",
        "checkpoint": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/converter/checkpoint.pth"
    },
    "base_speakers": {
        "EN": {
            "config": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/config.json",
            "checkpoint": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/checkpoint.pth",
            "default_se": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/en_default_se.pth",
            "style_se": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/EN/en_style_se.pth"
        },
        "ZH": {
            "config": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/ZH/config.json",
            "checkpoint": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/ZH/checkpoint.pth",
            "default_se": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/ZH/zh_default_se.pth",
            "style_se": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints/base_speakers/ZH/zh_style_se.pth"
        },
        # Add other languages here with similar structure
    }
}

# Cache for loaded models to avoid reloading
model_cache = {
    "base_speaker_tts": {},
    "tone_color_converter": None,
    "source_se": {}
}

def download_file(url, destination):
    """Download a file from a URL to a destination path with progress tracking"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Skip if file already exists
    if os.path.exists(destination):
        return True
    
    try:
        print(f"Downloading {url} to {destination}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(destination, 'wb') as file:
            for data in tqdm.tqdm(
                response.iter_content(block_size),
                total=total_size // block_size,
                unit='KB',
                unit_scale=True
            ):
                file.write(data)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_models(language_codes=None, progress=gr.Progress()):
    """Download model checkpoints for specified languages or all languages if None"""
    if language_codes is None:
        language_codes = list(LANGUAGES.values())
    
    # Make sure language_codes is a list
    if isinstance(language_codes, str):
        language_codes = [language_codes]
    
    total_steps = 1 + len(language_codes) * 4  # Converter + 4 files per language
    current_step = 0
    
    # Download converter
    progress(current_step / total_steps, desc="Downloading converter model...")
    download_file(MODEL_CONFIG["converter"]["config"], "checkpoints/converter/config.json")
    download_file(MODEL_CONFIG["converter"]["checkpoint"], "checkpoints/converter/checkpoint.pth")
    current_step += 1
    
    # Download base speakers for each language
    for lang_code in language_codes:
        if lang_code not in MODEL_CONFIG["base_speakers"]:
            print(f"No download info for language {lang_code}, skipping...")
            current_step += 4
            continue
        
        progress(current_step / total_steps, desc=f"Downloading {lang_code} config...")
        download_file(
            MODEL_CONFIG["base_speakers"][lang_code]["config"], 
            f"checkpoints/base_speakers/{lang_code}/config.json"
        )
        current_step += 1
        
        progress(current_step / total_steps, desc=f"Downloading {lang_code} checkpoint...")
        download_file(
            MODEL_CONFIG["base_speakers"][lang_code]["checkpoint"], 
            f"checkpoints/base_speakers/{lang_code}/checkpoint.pth"
        )
        current_step += 1
        
        progress(current_step / total_steps, desc=f"Downloading {lang_code} default embedding...")
        download_file(
            MODEL_CONFIG["base_speakers"][lang_code]["default_se"], 
            f"checkpoints/base_speakers/{lang_code}/{lang_code.lower()}_default_se.pth"
        )
        current_step += 1
        
        progress(current_step / total_steps, desc=f"Downloading {lang_code} style embedding...")
        download_file(
            MODEL_CONFIG["base_speakers"][lang_code]["style_se"], 
            f"checkpoints/base_speakers/{lang_code}/{lang_code.lower()}_style_se.pth"
        )
        current_step += 1
    
    progress(1.0, desc="Download complete!")
    return "Model download complete! You can now use OpenVoice."

def check_models_exist(language=None):
    """Check if models exist for specified language or any language if None"""
    # Check converter model
    if not os.path.exists('checkpoints/converter/checkpoint.pth'):
        return False
    
    if language:
        # Check specific language
        lang_code = LANGUAGES.get(language, language)
        return os.path.exists(f'checkpoints/base_speakers/{lang_code}/checkpoint.pth')
    else:
        # Check if any language model exists
        for lang_code in LANGUAGES.values():
            if os.path.exists(f'checkpoints/base_speakers/{lang_code}/checkpoint.pth'):
                return True
        return False

def load_tone_color_converter():
    """Load the tone color converter model"""
    if model_cache["tone_color_converter"] is not None:
        return model_cache["tone_color_converter"]
    
    ckpt_converter = 'checkpoints/converter'
    
    if not os.path.exists(f'{ckpt_converter}/checkpoint.pth'):
        raise FileNotFoundError(f"Converter model checkpoint not found at {ckpt_converter}/checkpoint.pth")
    
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
    
    # Cache the model
    model_cache["tone_color_converter"] = tone_color_converter
    
    return tone_color_converter

def load_base_speaker_tts(language):
    """Load the base speaker TTS model for the specified language"""
    lang_code = LANGUAGES.get(language, language)
    
    # Return from cache if available
    if lang_code in model_cache["base_speaker_tts"]:
        return model_cache["base_speaker_tts"][lang_code]
    
    ckpt_base = f'checkpoints/base_speakers/{lang_code}'
    
    if not os.path.exists(f'{ckpt_base}/checkpoint.pth'):
        raise FileNotFoundError(f"Base speaker model checkpoint not found at {ckpt_base}/checkpoint.pth")
    
    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
    
    # Cache the model
    model_cache["base_speaker_tts"][lang_code] = base_speaker_tts
    
    return base_speaker_tts

def get_source_embedding(language, style):
    """Get the source embedding for the specified language and style"""
    lang_code = LANGUAGES.get(language, language)
    style_code = STYLES.get(style, style)
    
    # Generate a cache key
    cache_key = f"{lang_code}_{style_code}"
    
    # Return from cache if available
    if cache_key in model_cache["source_se"]:
        return model_cache["source_se"][cache_key]
    
    # Determine the appropriate source embedding file
    if style_code == "default":
        source_path = f'checkpoints/base_speakers/{lang_code}/{lang_code.lower()}_default_se.pth'
    else:
        source_path = f'checkpoints/base_speakers/{lang_code}/{lang_code.lower()}_style_se.pth'
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source embedding file not found at {source_path}")
    
    source_se = torch.load(source_path).to(device)
    
    # Cache the embedding
    model_cache["source_se"][cache_key] = source_se
    
    return source_se

def process_reference_voice(reference_audio):
    """Process reference voice to extract tone color embedding"""
    # Load tone color converter
    tone_color_converter = load_tone_color_converter()
    
    # Process the reference voice
    target_se, audio_name = se_extractor.get_se(
        reference_audio, 
        tone_color_converter, 
        target_dir='processed', 
        vad=True
    )
    
    return target_se, audio_name

def clone_voice(text, reference_audio, language, style, speed=1.0, progress=gr.Progress()):
    """Generate speech with the reference voice and selected style"""
    try:
        # Validate inputs
        if not text.strip():
            return None, "Error: Please enter text to speak."
        
        if not reference_audio:
            return None, "Error: Please upload or record a reference voice."
        
        # Get language and style codes
        lang_code = LANGUAGES.get(language)
        style_code = STYLES.get(style)
        
        if not lang_code:
            return None, f"Error: Unsupported language '{language}'"
        
        if not style_code:
            return None, f"Error: Unsupported style '{style}'"
        
        # Check if models exist
        if not check_models_exist(lang_code):
            return None, f"Error: Models for {language} not found. Please download them first."
        
        progress(0.1, desc="Loading models...")
        
        # Load models and embeddings
        try:
            base_speaker_tts = load_base_speaker_tts(lang_code)
            tone_color_converter = load_tone_color_converter()
            source_se = get_source_embedding(lang_code, style_code)
        except FileNotFoundError as e:
            return None, f"Error: {str(e)}"
        
        progress(0.3, desc="Processing reference voice...")
        
        # Process reference voice
        try:
            target_se, audio_name = process_reference_voice(reference_audio)
        except Exception as e:
            return None, f"Error processing reference voice: {str(e)}"
        
        progress(0.6, desc="Generating base speech...")
        
        # Generate temporary audio with base speaker
        timestamp = int(torch.rand(1).item() * 100000)
        output_dir = 'outputs'
        src_path = f'{output_dir}/tmp_{timestamp}.wav'
        save_path = f'{output_dir}/output_{lang_code}_{style_code}_{timestamp}.wav'
        
        try:
            # Generate base speech
            base_speaker_tts.tts(
                text, 
                src_path, 
                speaker=style_code, 
                language=language, 
                speed=float(speed)
            )
        except Exception as e:
            return None, f"Error generating base speech: {str(e)}"
        
        progress(0.8, desc="Converting voice...")
        
        try:
            # Convert to target voice
            encode_message = "@MyShell"
            tone_color_converter.convert(
                audio_src_path=src_path, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=save_path,
                message=encode_message
            )
        except Exception as e:
            if os.path.exists(src_path):
                os.remove(src_path)
            return None, f"Error converting voice: {str(e)}"
        
        progress(1.0, desc="Done!")
        
        # Clean up the temporary file
        if os.path.exists(src_path):
            os.remove(src_path)
        
        return save_path, "Voice cloning completed successfully!"
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error: {str(e)}\n\nDetails: {error_details}"

# Create example texts for different languages
example_texts = {
    "English": "Hello, this is a voice generated using OpenVoice. The technology allows for accurate voice cloning with style control.",
    "Chinese": "你好，这是使用OpenVoice生成的声音。这项技术可以实现精确的声音克隆和风格控制。",
    "Spanish": "Hola, esta es una voz generada con OpenVoice. La tecnología permite una clonación precisa de voz con control de estilo.",
    "French": "Bonjour, c'est une voix générée en utilisant OpenVoice. La technologie permet un clonage vocal précis avec contrôle du style.",
    "Japanese": "こんにちは、これはOpenVoiceを使用して生成された音声です。この技術により、スタイル制御を備えた正確な音声クローンが可能になります。",
    "Korean": "안녕하세요, 이것은 OpenVoice를 사용하여 생성된 음성입니다. 이 기술은 스타일 제어를 통한 정확한 음성 복제를 가능하게 합니다."
}

def update_example_text(language):
    """Update the example text based on the selected language"""
    return example_texts.get(language, example_texts["English"])

# Create the Gradio interface
with gr.Blocks(title="OpenVoice - Voice Cloning") as demo:
    # Header
    with gr.Row():
        gr.Markdown(
            """
            # 🎙️ OpenVoice - Voice Cloning
            
            An advanced interface for MyShell AI's OpenVoice technology. Clone any voice with style control and multi-lingual support.
            """
        )
    
    # Model download section
    with gr.Row(visible=not check_models_exist()):
        with gr.Column():
            gr.Markdown(
                """
                ### ⚠️ Model Checkpoints Not Found
                
                Please download the necessary model checkpoints to use OpenVoice.
                """
            )
            with gr.Row():
                download_all_btn = gr.Button("Download All Models", variant="primary")
                with gr.Column():
                    lang_for_download = gr.Dropdown(
                        choices=list(LANGUAGES.keys()), 
                        label="Or select specific language to download",
                        value="English"
                    )
                    download_lang_btn = gr.Button("Download Selected Language")
            download_status = gr.Textbox(label="Download Status", interactive=False)
    
    # Main interface
    with gr.Tabs():
        with gr.TabItem("Voice Cloning"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Reference voice section
                    with gr.Group():
                        gr.Markdown("### 1. Upload Reference Voice")
                        gr.Markdown("*Upload or record a 3-10 second sample of the voice you want to clone*")
                        reference_audio = gr.Audio(
                            label="Voice to Clone", 
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                    
                    # Settings section
                    with gr.Group():
                        gr.Markdown("### 2. Configure Settings")
                        
                        # Language and style selection
                        with gr.Row():
                            language_dropdown = gr.Dropdown(
                                choices=list(LANGUAGES.keys()), 
                                label="Language", 
                                value="English"
                            )
                            style_dropdown = gr.Dropdown(
                                choices=list(STYLES.keys()), 
                                label="Voice Style", 
                                value="Default"
                            )
                        gr.Markdown("*Select the language and style for the generated speech*")
                        
                        # Speed control
                        speed_slider = gr.Slider(
                            minimum=0.5, 
                            maximum=1.5, 
                            value=1.0, 
                            label="Speech Speed", 
                            step=0.1
                        )
                        gr.Markdown("*Adjust the speed of the generated speech*")
                    
                    # Text input section
                    with gr.Group():
                        gr.Markdown("### 3. Enter Text")
                        text_input = gr.Textbox(
                            label="Text to Speak", 
                            lines=5, 
                            placeholder="Enter the text you want to convert to speech...",
                            value=example_texts["English"]
                        )
                        generate_btn = gr.Button("Generate Speech", variant="primary")
                
                with gr.Column(scale=1):
                    # Output section
                    with gr.Group():
                        gr.Markdown("### Generated Speech")
                        output_audio = gr.Audio(label="Output")
                        output_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Style examples section
                    with gr.Accordion("Voice Style Guide", open=False):
                        gr.Markdown("""
                        ### Voice Style Descriptions
                        
                        - **Default**: Neutral, balanced tone for general use
                        - **Friendly**: Warm, approachable tone with positive inflection
                        - **Cheerful**: Upbeat, happy tone with higher pitch variation
                        - **Excited**: Enthusiastic, energetic tone with quick pace
                        - **Sad**: Melancholic tone with lower pitch and slower pace
                        - **Angry**: Intense, forceful tone with strong articulation
                        - **Terrified**: Anxious tone with trembling quality
                        - **Shouting**: Loud, projected tone with increased volume
                        - **Whispering**: Hushed, quiet tone with reduced volume
                        
                        Choose the style that best fits your intended message and emotion.
                        """)
        
        with gr.TabItem("Instructions"):
            gr.Markdown("""
            ## How to Use OpenVoice
            
            OpenVoice is a versatile voice cloning system that lets you clone any voice and control its style in multiple languages.
            
            ### Getting Started
            
            1. **Download Models**: If you haven't already, download the models for your desired language(s).
            
            2. **Upload Reference Voice**:
               - Upload or record a 3-10 second audio clip of the voice you want to clone
               - For best results, use clear speech with minimal background noise
               - The reference voice can be in any language - OpenVoice can cross between languages
            
            3. **Configure Settings**:
               - **Language**: Select the output language for the generated speech
               - **Voice Style**: Choose a voice style (Default, Cheerful, Sad, etc.)
               - **Speech Speed**: Adjust the pace of the generated speech
            
            4. **Enter Text**:
               - Type the text you want to convert to speech
               - Make sure to use the appropriate characters for the selected language
            
            5. **Generate Speech**:
               - Click the "Generate Speech" button
               - Wait for the processing to complete (this may take a few seconds)
            
            ### Tips for Best Results
            
            - **Reference Audio Quality**: Use high-quality audio with clear speech
            - **Reference Length**: 3-10 seconds of speech works best
            - **Avoid Background Noise**: Minimize background noise in reference recordings
            - **Match Style to Content**: Select a style that matches your message content
            - **Language-Specific Characters**: Use proper characters and punctuation for each language
            - **Experiment**: Try different styles and speed settings for optimal results
            
            ### Troubleshooting
            
            - If generation fails, try using a different reference audio
            - Ensure your text is appropriate for the selected language
            - If a language model is missing, download it from the download tab
            """)
        
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About OpenVoice
            
            OpenVoice is a versatile instant voice cloning model developed by MyShell AI, capable of accurately cloning voices across languages and styles.
            
            ### OpenVoice V2 (April 2024)
            
            The latest version includes:
            
            - **Enhanced Audio Quality**: Improved training strategies for better sound quality
            - **Multi-lingual Support**: Native support for English, Spanish, French, Chinese, Japanese, and Korean
            - **MIT License**: Free for both commercial and research use
            
            ### Key Features
            
            - **Accurate Tone Color Cloning**: Clone voice characteristics while preserving the unique timbre
            - **Flexible Voice Style Control**: Modify emotions, accents, rhythm, pauses, and intonation
            - **Zero-shot Cross-lingual Voice Cloning**: Clone voices across languages without prior training
            
            ### How It Works
            
            OpenVoice uses a two-stage approach:
            
            1. A base speaker TTS model generates speech with the desired style and language
            2. A tone color converter transforms this speech to match the reference voice
            
            This separation allows for both accurate voice cloning and flexible style control.
            
            ### Research
            
            For more technical details, see the [research paper](https://arxiv.org/abs/2312.01479):
            
            ```
            @article{qin2023openvoice,
              title={OpenVoice: Versatile Instant Voice Cloning},
              author={Qin, Zengyi and Zhao, Wenliang and Yu, Xumin and Sun, Xin},
              journal={arXiv preprint arXiv:2312.01479},
              year={2023}
            }
            ```
            
            ### Links
            
            - [GitHub Repository](https://github.com/myshell-ai/OpenVoice)
            - [Project Website](https://research.myshell.ai/open-voice)
            - [MyShell AI](https://myshell.ai)
            """)
    
    # Set up event handlers
    download_all_btn.click(
        lambda: download_models(), 
        outputs=download_status
    )
    
    download_lang_btn.click(
        lambda lang: download_models(LANGUAGES[lang]), 
        inputs=lang_for_download,
        outputs=download_status
    )
    
    language_dropdown.change(
        update_example_text,
        inputs=language_dropdown,
        outputs=text_input
    )
    
    generate_btn.click(
        clone_voice, 
        inputs=[text_input, reference_audio, language_dropdown, style_dropdown, speed_slider], 
        outputs=[output_audio, output_status]
    )

# Launch the app
if __name__ == "__main__":
    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Launch the interface
    demo.launch(share=False)