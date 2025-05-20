import os
import re
import openai
import discord
from discord.ext import commands
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import time
from collections import defaultdict
import openai.error

# --------- API KEYS (use .env or Render secrets in production) ---------
openai.api_key = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --------- Config ---------
INDEX_FILE = "sop_index.faiss"
META_FILE = "sop_chunks.pkl"
MODEL_NAME = "gpt-3.5-turbo"
COOLDOWN_SECONDS = 10

# --------- Load FAISS Index & Metadata ---------
if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
    raise FileNotFoundError("❌ Missing FAISS index or metadata file.")

index = faiss.read_index(INDEX_FILE)
with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

# --------- Embedding Model ---------
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# --------- Discord Setup ---------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --------- Cooldown Tracker ---------
user_last_asked = defaultdict(lambda: 0)

@bot.event
async def on_ready():
    print(f"✅ {bot.user.name} is online!")

@bot.command()
async def ping(ctx):
    try:
        await ctx.send("✅ I'm alive!")
    except discord.errors.HTTPException as e:
        print(f"⚠️ Failed to send ping response: {str(e)}")

def split_message(text, limit=2000):
    lines = text.split('\n')
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 <= limit:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) and "?" in message.content:
        now = time.time()
        user_id = message.author.id

        if now - user_last_asked[user_id] < COOLDOWN_SECONDS:
            try:
                await message.channel.send(
                    f"⏳ {message.author.mention}, please wait a few seconds before asking again."
                )
            except discord.errors.HTTPException:
                print("⚠️ Could not send cooldown warning due to rate limit.")
            return

        user_last_asked[user_id] = now
        user_question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        try:
            # Step 1: Embed user question
            q_embedding = EMBEDDING_MODEL.encode([user_question])
            D, I = index.search(np.array(q_embedding).astype("float32"), k=8)

            # Step 2: Build clean context from top matches
            max_context_chars = 7000
            context_parts = []
            for i in I[0]:
                if 0 <= i < len(metadata):
                    excerpt = metadata[i]["content"]
                    if len(excerpt.strip()) > 50:
                        context_parts.append(f"- {excerpt.strip()}")

            if not context_parts:
                await message.channel.send(
                    f"📄 {message.author.mention} Sorry, I couldn’t find anything in the SOP to help with that. "
                    "Please contact your manager or refer to AGCO guidelines for clarification."
                )
                return

            context = "\n".join(context_parts)[:max_context_chars]

            # Step 3: Construct OpenAI message
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
                        "If a user asks about Thomas Kitchens, the weather, or what to do with their spare time, respond in a playful and lighthearted tone using the jokes and ideas provided in the SOP content."
                    )
                },
                {
                    "role": "user",
                    "content": f"Relevant SOP Excerpts:\n{context}\n\nUser Question: {user_question}"
                }
            ]

            # Step 4: Query OpenAI with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=600,
                        temperature=0.5
                    )
                    break  # success
                except openai.error.RateLimitError:
                    wait_time = 2 ** attempt
                    print(f"⚠️ OpenAI rate limit hit. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    print(f"❌ OpenAI API error: {e}")
                    await message.channel.send("⚠️ OpenAI error occurred. Please try again later.")
                    return
            else:
                await message.channel.send("⚠️ OpenAI is currently overloaded. Please try again later.")
                return

            answer = response.choices[0].message.content.strip()

            # Step 5: Send reply with throttling and retry-on-429 logic
            for chunk in split_message(f"🧠 {message.author.mention} {answer}"):
                try:
                    await message.channel.send(chunk)
                    await asyncio.sleep(1)
                except discord.errors.HTTPException as e:
                    if e.status == 429:
                        retry_after = None
                        if e.response and hasattr(e.response, "headers"):
                            retry_after = e.response.headers.get("Retry-After")
                        try:
                            wait_time = float(retry_after) if retry_after else 5.0
                            print(f"⚠️ Rate limit hit. Waiting {wait_time:.2f} seconds before retrying...")
                            await asyncio.sleep(wait_time)
                            await message.channel.send(chunk)
                            await asyncio.sleep(1)
                        except Exception as retry_err:
                            print(f"⚠️ Retry failed or invalid wait time: {retry_err}")
                            break
                    else:
                        print(f"⚠️ Error sending message: {str(e)}")
                        break

        except discord.errors.HTTPException as e:
            if e.status == 429:
                print("⚠️ Rate limited. No error message sent.")
            else:
                await message.channel.send("⚠️ Something went wrong with Discord.")
            print(f"Discord Error: {str(e)}")

        except Exception as e:
            await message.channel.send("⚠️ Something went wrong while processing your request.")
            print(f"Error: {str(e)}")

    await bot.process_commands(message)

# --------- Run Bot ---------
bot.run(DISCORD_TOKEN)


