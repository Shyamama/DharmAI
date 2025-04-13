# app.py - Loads ONLY the pre-trained model for generation (Corrected)



import torch

from transformers import (

    AutoModelForCausalLM,

    AutoTokenizer,

    BitsAndBytesConfig, # Keep for 4-bit loading efficiency

    pipeline

)

import warnings

import readline # Optional: Improves CLI input experience

import os # Needed for debugging path checks



# --- Configuration ---

# Load the model you created with pretrain.py (Relative Path)

model_id = "microsoft/phi-2"

# Attempt to use GPU if available, otherwise fallback (will be slow)

device = "cuda" if torch.cuda.is_available() else "cpu"



# Suppress warnings

warnings.filterwarnings("ignore")



print(f"--- Scripture Text Generator (Pre-trained Model) ---")

print(f"Using device: {device}")



# --- Load Model Components ---



# 1. Quantization Config (Corrected)

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    load_in_8bit=False,  # Explicitly set to False

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16 # Correct data type

)

print(f"BitsAndBytesConfig created.") # Added print



# 2. Load Tokenizer (from your pre-trained directory)

try:

    # --- ADD DEBUGGING ---

    print(f"\nDEBUG: Current directory is: {os.getcwd()}")

    print(f"DEBUG: Attempting to load tokenizer from model_id: '{model_id}'")

    # Check if directory exists right before loading

    full_path_check = os.path.join(os.getcwd(), model_id)

    print(f"DEBUG: Checking existence of full path: '{full_path_check}'")

    print(f"DEBUG: Does path '{model_id}' exist here? {os.path.isdir(model_id)}") # Relative check



    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Tokenizer loaded successfully from {model_id}.") # Added success print

    if tokenizer.pad_token is None:

        # Set pad token for generation if not present

        tokenizer.pad_token = tokenizer.eos_token

        print("Set pad_token to eos_token")

except Exception as e:

    print(f"ERROR: Could not load tokenizer from {model_id}. Make sure the directory exists and contains tokenizer files.")

    print(e)

    exit()



# 3. Load Pre-trained Model (NO LoRA/PEFT here)

try:

    # --- ADD DEBUGGING ---

    print(f"\nDEBUG: Current directory is: {os.getcwd()}")

    print(f"DEBUG: Attempting to load model from model_id: '{model_id}'")

    print(f"DEBUG: Does path '{model_id}' exist here? {os.path.isdir(model_id)}") # Relative check

     # ---------------------

    model = AutoModelForCausalLM.from_pretrained(

        model_id,

        quantization_config=bnb_config, # Use quantization

        device_map="auto",           # Automatically map layers (tries GPU first)

        trust_remote_code=True       # May be needed depending on the base model used (like Phi-2)

    )

    print(f"Model loaded successfully from {model_id}.") # Added success print

except Exception as e:

    print(f"ERROR: Could not load model from {model_id}. Make sure the directory exists and contains model files (like pytorch_model.bin or shards).")

    print(e)

    exit()





# Set model to evaluation mode (disables dropout etc.)

model.eval()

print("Model set to evaluation mode.")



# --- Create Inference Pipeline ---

try:

    print("Creating text generation pipeline...")

    # Ensure the pipeline uses the same device mapping

    pipe = pipeline(

        "text-generation",

        model=model,

        tokenizer=tokenizer,

        device_map="auto"

    )

    print("✅ Model and pipeline ready!")

except Exception as e:

    print(f"ERROR: Could not create text generation pipeline.")

    print(e)

    exit()



# --- Interaction Loop ---

print("\nEnter a prompt (e.g., a starting verse, a topic like 'Karma is...')")

print('Type "quit" to exit.')



while True:

    try:

        prompt = input("\nPrompt: ")

        if prompt.lower() == "quit":

            break



        if not prompt:

            continue



        print("Generating text...")



        # Generate text using the pipeline

        # Adjust parameters as needed

        sequences = pipe(

            prompt,

            max_new_tokens=150,       # Generate up to 150 new tokens

            do_sample=True,          # Use sampling for more creative output

            temperature=0.7,         # Controls randomness (lower = more focused)

            top_k=50,                # Consider top K tokens

            top_p=0.95,              # Use nucleus sampling

            num_return_sequences=1,

            eos_token_id=tokenizer.eos_token_id,

            pad_token_id=tokenizer.pad_token_id # Use pad token ID for padding during generation

        )



        # Print the generated text (includes the prompt by default)

        generated_text = sequences[0]['generated_text']

        print("\nModel Output:\n", generated_text)



    except EOFError: # Handle Ctrl+D

        break

    except KeyboardInterrupt: # Handle Ctrl+C

        break

    except Exception as e:

        print(f"\nAn error occurred during generation: {e}")

        # Continue the loop

        # print("Please try again.")



print("\nExiting generator. Goodbye!")
