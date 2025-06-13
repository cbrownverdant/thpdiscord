import os
import openai
import discord
from discord.ext import commands, tasks
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import time
from collections import defaultdict
import openai.error
import gc
import psutil


# --------- API KEYS (use .env or Render secrets in production) ---------
openai.api_key = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --------- Config ---------
INDEX_FILE = "sop_index.faiss"
META_FILE = "sop_chunks.pkl"
MODEL_NAME = "gpt-3.5-turbo"
COOLDOWN_SECONDS = 10
MAX_TOKENS = 1000
MEMORY_THRESHOLD_MB = 400 

# --------- Lazy Load Variables ---------
index = None
metadata = None
EMBEDDING_MODEL = None
user_last_asked = defaultdict(lambda: 0)

# --------- Discord Setup ---------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, max_messages=0)

# --------- Memory Management ---------
def log_memory_usage(label=""):
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"[{time.ctime()}] 🧠 {label} - RSS: {mem_mb:.2f} MB")
    return mem_mb

def maybe_unload_memory(force=False):
    global index, metadata, EMBEDDING_MODEL
    mem = log_memory_usage("Memory Check")
    if force or mem > MEMORY_THRESHOLD_MB:
        print("⚠️ Memory exceeded threshold. Releasing memory.")
        index = metadata = EMBEDDING_MODEL = None
        gc.collect()
        log_memory_usage("After memory cleanup")

@tasks.loop(minutes=10)
async def memory_logger_task():
    maybe_unload_memory()

async def auto_restart_timer():
    await asyncio.sleep(6 * 60 * 60)  # Restart every 6 hours
    print("🔁 Auto-restarting to avoid memory leaks...")
    os._exit(0)

# --------- Helpers ---------
async def safe_send(channel, content):
    try:
        await channel.send(content)
    except discord.errors.HTTPException as e:
        retry_after = float(e.response.headers.get("Retry-After", 5.0)) if e.response else 5.0
        print(f"🚦 Discord 429 hit. Retrying in {retry_after:.2f}s...")
        await asyncio.sleep(retry_after)
        await channel.send(content)

def split_message(text, limit=2000):
    lines, chunks, current = text.split('\n'), [], ""
    for line in lines:
        if len(current) + len(line) + 1 <= limit:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

# --------- Events ---------
@bot.event
async def on_ready():
    print(f"✅ {bot.user.name} is online!")
    if not memory_logger_task.is_running():
        memory_logger_task.start()
    bot.loop.create_task(auto_restart_timer())


@bot.event
async def on_message(message):
    global index, metadata, EMBEDDING_MODEL

    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) and "?" in message.content:
        now = time.time()
        user_id = message.author.id

        if now - user_last_asked[user_id] < COOLDOWN_SECONDS:
            await safe_send(message.channel, f"⏳ {message.author.mention}, please wait a few seconds before asking again.")
            return

        user_last_asked[user_id] = now
        user_question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        try:
            # Load FAISS index if not already loaded
            if index is None:
                if not os.path.exists(INDEX_FILE):
                    raise FileNotFoundError("❌ FAISS index missing.")
                index = faiss.read_index(INDEX_FILE)

            # Load metadata
            if metadata is None:
                if not os.path.exists(META_FILE):
                    raise FileNotFoundError("❌ Metadata pickle missing.")
                with open(META_FILE, "rb") as f:
                    metadata = pickle.load(f)

            # Load embedding model only if not in memory
            if EMBEDDING_MODEL is None:
                EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

            # Encode the question
            q_embedding = await asyncio.to_thread(EMBEDDING_MODEL.encode, [user_question])
            D, I = index.search(np.array(q_embedding, dtype="float32"), k=5)

            context_parts = []
            for i in I[0]:
                if 0 <= i < len(metadata):
                    content = metadata[i]['content'].strip()
                    if len(content) > 50:
                        context_parts.append(f"- {content}")

            if not context_parts:
                await safe_send(message.channel, f"📄 {message.author.mention} Sorry, I couldn’t find anything in the SOP.")
                return

            context = "\n".join(context_parts)[:7000]

            # Compose message for OpenAI
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for staff at The Hunny Pot Cannabis Co. "
                        "Only answer questions using the SOP excerpts provided below. "
                        "Provide only what is relevant to answer the user's question. Do not include unrelated or extra information. "
                        "If the SOP contains a step-by-step procedure, explain the process clearly in natural language. "
                        "When helpful for clarity, organize multi-step procedures into clear bullet points. "
                        "If the SOP includes multiple procedures related to the same task (e.g., system steps in Cova and warehouse/packaging steps), label each set of steps accordingly so the user understands the different parts of the process. "
                        "Do not copy the original SOP's formatting — summarize steps in your own words, clearly and practically. "
                        "If a link (URL) is present in the SOP excerpt and it's relevant to the question, always include it exactly as written. "
                        "Never invent or modify URLs. Only share links that appear in the SOP excerpts. "
                        "If the SOP does not contain an answer, say so directly and do not attempt to guess. "
                        "If names, phone numbers, or emails are shown in the SOP excerpts, include them only as written. "
                        "Do not mention SOP file names, document numbers, or section titles. "
                        "Never say things like 'according to the SOP' — just provide the correct procedure or information clearly and practically. "
                        "If a user asks about Thomas Kitchens, the weather, or what to do with their spare time, respond in a playful and lighthearted tone using the jokes and ideas provided in the SOP content. "
                        "Feel free to include a light joke when appropriate, as long as it doesn’t affect the accuracy or clarity of the response. If a user explicitly asks for a joke, respond with one accordingly."
                    )

                },

                {
                    "role": "user",
                    "content": f"Relevant SOP Excerpts:\n{context}\n\nUser Question: {user_question}"
                }
            ]

            # Send to OpenAI with retry
            for attempt in range(3):
                try:
                    response = openai.ChatCompletion.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=MAX_TOKENS,
                        temperature=0.5
                    )
                    break
                except openai.error.RateLimitError:
                    wait = 2 ** attempt
                    print(f"⚠️ OpenAI rate limit hit. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                except Exception as e:
                    print(f"❌ OpenAI error: {e}")
                    await safe_send(message.channel, "⚠️ OpenAI error occurred.")
                    return
            else:
                await safe_send(message.channel, "⚠️ OpenAI is overloaded. Please try again later.")
                return

            answer = response.choices[0].message.content.strip()
            full_response = f"🧠 {message.author.mention} {answer}"

            for chunk in split_message(full_response):
                await safe_send(message.channel, chunk)
                await asyncio.sleep(1)

            # Clean up
            del q_embedding, D, I, context_parts, context, messages, response, answer
            gc.collect()

            # Unload model only if needed
            if log_memory_usage("Post-response check") > MEMORY_THRESHOLD_MB:
                EMBEDDING_MODEL = None
                gc.collect()

            maybe_unload_memory()

        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            await safe_send(message.channel, "⚠️ Something went wrong.")

    await bot.process_commands(message)

# --------- Run Bot ---------
bot.run(DISCORD_TOKEN)


